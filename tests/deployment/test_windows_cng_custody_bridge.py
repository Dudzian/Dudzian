"""TEST_ONLY ABI simulation; Windows and physical TPM qualification are NOT_RUN."""

from __future__ import annotations

import ctypes
import hashlib
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils

from bot_core.licensing.device_enrollment import TPMPublicProjectionV1, make_evidence
from bot_core.licensing.pre_enrollment import PreEnrollmentRequestV1
from bot_core.licensing.production_tpm_custody import (
    K_PSA_PROFILE,
    PROJECTION_PROFILE,
    PROJECTION_SOURCE_PROFILE,
    ProductionTPMCustodyError,
    pre_enrollment_custody_qualifying_data,
)
from deployment import windows_cng_custody_bridge as bridge, windows_cng_pre_enrollment as cng
from tests.deployment.test_windows_cng_pre_enrollment import (
    TestOnlyFunction,
    TestOnlyNCryptDLL,
    request_for,
    set_number,
    wide,
)


def two_b(raw: bytes) -> bytes:
    return struct.pack(">H", len(raw)) + raw


def sec1(private: ec.EllipticCurvePrivateKey) -> bytes:
    return private.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )


def public_area(public: bytes, *, role: str) -> bytes:
    attributes = {"subject": 0x40072, "ak": 0x50072, "k_psa": 0x400B2, "ek": 0x300B2}[role]
    policy = {
        "subject": b"",
        "ak": b"",
        "k_psa": b"p" * 32,
        "ek": bytes.fromhex("837197674484b3f81a90cc8d46a5d724fd52d76e06520b64f2a1da1b331469aa"),
    }[role]
    symmetric = b"\x00\x06\x00\x80\x00\x43" if role == "ek" else b"\x00\x10"
    scheme = b"\x00\x10" if role == "ek" else b"\x00\x18\x00\x0b"
    return (
        struct.pack(">HHI", 0x23, 0x0B, attributes)
        + two_b(policy)
        + symmetric
        + scheme
        + b"\x00\x03\x00\x10"
        + two_b(public[1:33])
        + two_b(public[33:])
    )


def name_of(raw: bytes) -> bytes:
    return b"\x00\x0b" + hashlib.sha256(raw).digest()


def response(parameters: bytes, *, auth: bool) -> bytes:
    body = (
        struct.pack(">I", len(parameters)) + parameters + bridge._PW_RESPONSE
        if auth
        else parameters
    )
    return struct.pack(">HII", 0x8002 if auth else 0x8001, 10 + len(body), 0) + body


class TestOnlyTBSDLL:
    __test__ = False

    def __init__(self, subject: bytes, ak: bytes, private: ec.EllipticCurvePrivateKey) -> None:
        self.subject, self.ak, self.private = subject, ak, private
        self.context = 0x12345678
        self.commands: list[tuple[int, bytes]] = []
        self.overrides: dict[int, bytes] = {}
        self.status = 0
        self.reported_size: int | None = None
        self.device_status = 0
        self.device_version = 2
        self.device_structure_version = 1
        self.attest_override: bytes | None = None
        self.signature_override: bytes | None = None
        self.qualified_name = b"\x00\x0b" + b"q" * 32
        self.Tbsip_Submit_Command = TestOnlyFunction(self._submit)
        self.Tbsi_GetDeviceInfo = TestOnlyFunction(self._info)

    def _info(self, size: int, output: object) -> int:
        assert size == 16
        values = (ctypes.c_uint32 * 4)(self.device_structure_version, self.device_version, 0, 0)
        ctypes.memmove(output, values, size)
        return self.device_status

    def _submit(
        self,
        context: int,
        locality: int,
        priority: int,
        source: object,
        size: int,
        output: object,
        output_size: object,
    ) -> int:
        assert context == self.context
        assert (locality, priority) == (0, 200)
        command = ctypes.string_at(source, size)
        assert int.from_bytes(command[2:6], "big") == size
        code = int.from_bytes(command[6:10], "big")
        self.commands.append((context, command))
        if self.status:
            return self.status
        if code in self.overrides:
            raw = self.overrides[code]
        elif code == bridge.TPM_CC_READ_PUBLIC:
            handle = int.from_bytes(command[10:14], "big")
            assert command[:2] == b"\x80\x01" and len(command) == 14
            assert handle in {0x80000001, 0x80000002}
            public = self.subject if handle == 0x80000001 else self.ak
            raw = response(
                two_b(public) + two_b(name_of(public)) + two_b(self.qualified_name), auth=False
            )
        else:
            assert code == bridge.TPM_CC_CERTIFY_CREATION
            assert command[:2] == b"\x80\x02"
            assert command[10:18] == struct.pack(">II", 0x80000002, 0x80000001)
            assert command[18:31] == struct.pack(">II", 9, bridge.TPM_RS_PW) + bytes(5)
            qualifier = command[33:65]
            creation_hash = command[67:99]
            assert command[31:33] == command[65:67] == b"\x00\x20"
            assert command[99:103] == b"\x00\x18\x00\x0b"
            attest = (
                struct.pack(">IH", 0xFF544347, 0x801A)
                + two_b(self.qualified_name)
                + two_b(qualifier)
                + bytes(16)
                + b"\x01"
                + bytes(8)
                + two_b(name_of(self.subject))
                + two_b(creation_hash)
            )
            attest = self.attest_override if self.attest_override is not None else attest
            der = self.private.sign(
                hashlib.sha256(attest).digest(), ec.ECDSA(utils.Prehashed(hashes.SHA256()))
            )
            r, s = utils.decode_dss_signature(der)
            signature = (
                b"\x00\x18\x00\x0b" + two_b(r.to_bytes(32, "big")) + two_b(s.to_bytes(32, "big"))
            )
            signature = (
                self.signature_override if self.signature_override is not None else signature
            )
            raw = response(two_b(attest) + signature, auth=True)
        assert len(raw) <= bridge.MAX_RESPONSE
        ctypes.memmove(output, raw, len(raw))
        set_number(output_size, len(raw) if self.reported_size is None else self.reported_size)
        return 0


@dataclass
class TestOnlyCollection:
    __test__ = False
    key: cng.WindowsCNGPreEnrollmentKey
    request: PreEnrollmentRequestV1
    projection: TPMPublicProjectionV1
    ncrypt: TestOnlyNCryptDLL
    tbs: TestOnlyTBSDLL

    def collect(self) -> object:
        return bridge.certify_pre_enrollment_key_creation(
            self.key,
            self.request,
            ak_key_name="Stage9.Retained.AK",
            target_tpm_projection=self.projection,
        )


@pytest.fixture
def collection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestOnlyCollection]:
    simulation = TestOnlyNCryptDLL()
    native = object.__new__(cng._NCryptAPI)
    native.dll = simulation
    monkeypatch.setattr(cng, "_load_native", lambda: native)
    ak_private = ec.generate_private_key(ec.SECP256R1())
    subject = public_area(sec1(simulation.private), role="subject")
    ak = public_area(sec1(ak_private), role="ak")
    tbs = TestOnlyTBSDLL(subject, ak, ak_private)
    native_tbs = object.__new__(bridge._TBSCustodyAPI)
    native_tbs.dll = tbs
    monkeypatch.setattr(bridge, "_load_tbs_native", lambda: native_tbs)
    simulation.properties.update(
        {
            (11, "PCP_PLATFORMHANDLE"): tbs.context.to_bytes(
                ctypes.sizeof(ctypes.c_void_p), "little"
            ),
            (22, "PCP_PLATFORMHANDLE"): struct.pack("<I", 0x80000001),
            (22, "PCP_KEY_CREATIONHASH"): b"c" * 32,
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
    original_open = simulation.NCryptOpenKey.function

    def open_key(provider: int, output: object, name: str, legacy: int, flags: int) -> int:
        if name != "Stage9.Retained.AK":
            return original_open(provider, output, name, legacy, flags)
        simulation.calls.append(("open_ak", provider, name, legacy, flags))
        set_number(output, 33, ctypes.c_size_t)
        return 0

    simulation.NCryptOpenKey.function = open_key
    with cng.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path) as key:
        k_psa = public_area(sec1(ec.generate_private_key(ec.SECP256R1())), role="k_psa")
        ek = public_area(sec1(ec.generate_private_key(ec.SECP256R1())), role="ek")
        projection = make_evidence(
            public_area_hex=k_psa.hex(),
            returned_name=name_of(k_psa).hex(),
            creation_hash="63" * 32,
            ek_public_area_hex=ek.hex(),
            ek_name=name_of(ek).hex(),
            ak_public_area_hex=ak.hex(),
            ak_name=name_of(ak).hex(),
            algorithm_profile=K_PSA_PROFILE,
            evidence_profile=PROJECTION_PROFILE,
            substrate_profile=PROJECTION_SOURCE_PROFILE,
        )
        document = request_for(key.public_key_bytes).document
        document.update(
            {
                "verified_tpm_public_projection_id": projection.evidence_reference,
                "ek_public_digest": projection.document["ek"]["public_digest"],
                "ak_public_digest": projection.document["ak"]["public_digest"],
            }
        )
        simulation.calls.clear()
        yield TestOnlyCollection(
            key, PreEnrollmentRequestV1.from_mapping(document), projection, simulation, tbs
        )


@pytest.mark.parametrize("framed", [False, True])
def test_exact_existing_key_and_retained_ak_same_borrowed_context(
    collection: TestOnlyCollection, framed: bool
) -> None:
    if framed:
        collection.ncrypt.properties[(22, "PCP_KEY_CREATIONHASH")] = two_b(b"c" * 32)
    evidence = collection.collect()
    assert (
        evidence.document["pre_enrollment_request_digest_sha256"]
        == collection.request.digest_sha256
    )
    assert evidence.document["subject_creation_hash"] == (b"c" * 32).hex()
    assert not hasattr(evidence, "authenticated")
    assert len(collection.tbs.commands) == 3
    assert all(context == collection.tbs.context for context, _ in collection.tbs.commands)
    assert collection.tbs.commands[-1][1][33:65] == pre_enrollment_custody_qualifying_data(
        collection.request
    )
    assert ("open_ak", 11, "Stage9.Retained.AK", 0, cng.MACHINE_KEY) in collection.ncrypt.calls
    assert ("free", 33) in collection.ncrypt.calls
    assert not any(
        call[0] in {"create", "sign", "set", "delete"} for call in collection.ncrypt.calls
    )
    assert not any(
        call == ("free", value)
        for call in collection.ncrypt.calls
        for value in (22, collection.tbs.context, 0x80000001, 0x80000002)
    )


@pytest.mark.parametrize(
    "name,raw,error",
    [
        ("PCP_KEY_CREATIONHASH", b"x" * 31, "CREATION_HASH_CODEC"),
        ("PCP_KEY_CREATIONHASH", b"\x20\x00" + b"x" * 32, "CREATION_HASH_CODEC"),
        ("PCP_KEY_CREATIONHASH", two_b(b"x" * 32) + b"x", "CREATION_HASH_CODEC"),
        (
            "PCP_KEY_CREATIONTICKET",
            struct.pack(">HI", 0x8022, 0x40000001) + two_b(b"t" * 32),
            "CREATION_TICKET_CODEC",
        ),
        (
            "PCP_KEY_CREATIONTICKET",
            struct.pack(">HI", 0x8021, 0x40000007) + two_b(b"t" * 32),
            "CREATION_TICKET_CODEC",
        ),
        (
            "PCP_KEY_CREATIONTICKET",
            struct.pack(">HI", 0x8021, 0x40000001) + two_b(b"t" * 32) + b"x",
            "CREATION_TICKET_CODEC",
        ),
    ],
)
def test_unsupported_creation_codecs_fail_before_ak_or_tpm(
    collection: TestOnlyCollection, name: str, raw: bytes, error: str
) -> None:
    collection.ncrypt.properties[(22, name)] = raw
    with pytest.raises(bridge.WindowsCNGCustodyBridgeError, match=error):
        collection.collect()
    assert not collection.tbs.commands
    assert not any(call[0] == "open_ak" for call in collection.ncrypt.calls)


def test_unavailable_creation_property_never_creates_substitute(
    collection: TestOnlyCollection,
) -> None:
    collection.ncrypt.property_error = "PCP_KEY_CREATIONTICKET"
    with pytest.raises(
        bridge.WindowsCNGCustodyBridgeError, match="PCP_CREATION_EVIDENCE_UNAVAILABLE"
    ):
        collection.collect()
    assert not collection.tbs.commands
    assert not any(call[0] in {"create", "sign", "set"} for call in collection.ncrypt.calls)


@pytest.mark.parametrize("handle,width", [(11, 4), (22, 8)])
def test_provider_and_virtual_key_handle_abi_widths(
    collection: TestOnlyCollection, handle: int, width: int
) -> None:
    if handle == 11 and ctypes.sizeof(ctypes.c_void_p) == 4:
        width = 8
    collection.ncrypt.properties[(handle, "PCP_PLATFORMHANDLE")] = (1).to_bytes(width, "little")
    with pytest.raises(bridge.WindowsCNGCustodyBridgeError, match="PCP_PLATFORM_HANDLE"):
        collection.collect()
    assert not collection.tbs.commands


@pytest.mark.parametrize("raw", [bytes(4), struct.pack("<I", 0x81000001)])
def test_invalid_virtual_key_handle_rejected(collection: TestOnlyCollection, raw: bytes) -> None:
    collection.ncrypt.properties[(22, "PCP_PLATFORMHANDLE")] = raw
    with pytest.raises(bridge.WindowsCNGCustodyBridgeError, match="PCP_PLATFORM_HANDLE"):
        collection.collect()


@pytest.mark.parametrize(
    "field", ["verified_tpm_public_projection_id", "ak_public_digest", "ek_public_digest"]
)
def test_changed_request_projection_binding_rejected(
    collection: TestOnlyCollection, field: str
) -> None:
    document = collection.request.document
    document[field] = "b" * 64
    collection.request = PreEnrollmentRequestV1.from_mapping(document)
    with pytest.raises(
        bridge.WindowsCNGCustodyBridgeError, match="REQUEST_TPM_PROJECTION_MISMATCH"
    ):
        collection.collect()
    assert not collection.tbs.commands


def test_copied_key_cannot_cross_factory_guard(collection: TestOnlyCollection) -> None:
    forged = object.__new__(cng.WindowsCNGPreEnrollmentKey)
    for field in cng.WindowsCNGPreEnrollmentKey.__slots__:
        if field != "__weakref__":
            setattr(forged, field, getattr(collection.key, field))
    collection.key = forged
    with pytest.raises(
        cng.WindowsCNGPreEnrollmentError, match="VERIFIED_PRODUCTION_CNG_KEY_REQUIRED"
    ):
        collection.collect()
    assert not collection.tbs.commands


def test_closed_key_rejected(collection: TestOnlyCollection) -> None:
    collection.key.close()
    with pytest.raises(
        cng.WindowsCNGPreEnrollmentError, match="VERIFIED_PRODUCTION_CNG_KEY_REQUIRED"
    ):
        collection.collect()
    assert not collection.tbs.commands


@pytest.mark.parametrize(
    "name,raw",
    [
        ("PCP_PASSWORD_REQUIRED", b"\x01"),
        ("PCP_KEY_USAGE_POLICY", struct.pack("<I", 1)),
        ("Key Type", bytes(4)),
        ("PCP_EXPORT_ALLOWED", b"\x01"),
    ],
)
def test_unsupported_ak_auth_or_profile_never_certifies(
    collection: TestOnlyCollection, name: str, raw: bytes
) -> None:
    collection.ncrypt.properties[(33, name)] = raw
    with pytest.raises(
        bridge.WindowsCNGCustodyBridgeError, match="RETAINED_AK_PROFILE_UNSUPPORTED"
    ):
        collection.collect()
    assert not collection.tbs.commands
    assert ("free", 33) in collection.ncrypt.calls


@pytest.mark.parametrize("role", ["ak", "subject"])
def test_changed_tpm_public_never_certifies(collection: TestOnlyCollection, role: str) -> None:
    altered = public_area(sec1(ec.generate_private_key(ec.SECP256R1())), role=role)
    setattr(collection.tbs, role, altered)
    with pytest.raises(
        bridge.WindowsCNGCustodyBridgeError, match="AK_IDENTITY|TPM_PUBLIC_IDENTITY"
    ):
        collection.collect()
    assert all(
        int.from_bytes(command[6:10], "big") == bridge.TPM_CC_READ_PUBLIC
        for _, command in collection.tbs.commands
    )
    assert ("free", 33) in collection.ncrypt.calls


@pytest.mark.parametrize(
    "parameters",
    [
        b"",
        two_b(b"x"),
        two_b(b"x") + two_b(b"x") + two_b(b"x"),
        two_b(b"x") + two_b(b"x") + two_b(b"\x00\x0b" + b"q" * 32) + b"extra",
    ],
)
def test_malformed_read_public_fail_closed(
    collection: TestOnlyCollection, parameters: bytes
) -> None:
    collection.tbs.overrides[bridge.TPM_CC_READ_PUBLIC] = response(parameters, auth=False)
    with pytest.raises(bridge.WindowsCNGCustodyBridgeError):
        collection.collect()
    assert ("free", 33) in collection.ncrypt.calls


@pytest.mark.parametrize("status", [0x80284001, 0x80284005])
def test_tbs_failure_no_retry_and_ak_reference_freed(
    collection: TestOnlyCollection, status: int
) -> None:
    collection.tbs.status = status
    with pytest.raises(bridge.WindowsCNGCustodyBridgeError, match="TBS_SUBMIT_FAILED"):
        collection.collect()
    assert len(collection.tbs.commands) == 1
    assert ("free", 33) in collection.ncrypt.calls


@pytest.mark.parametrize("size", [0, 9, bridge.MAX_RESPONSE + 1])
def test_tbs_response_capacity_checked(collection: TestOnlyCollection, size: int) -> None:
    collection.tbs.reported_size = size
    with pytest.raises(bridge.WindowsCNGCustodyBridgeError, match="INVALID_TBS_RESPONSE_SIZE"):
        collection.collect()


@pytest.mark.parametrize(
    "raw,error",
    [
        (struct.pack(">HII", 0x8001, 10, 0x98E), "TPM_COMMAND_REJECTED"),
        (struct.pack(">HII", 0x8001, 11, 0), "TPM_RESPONSE_SIZE_MISMATCH"),
        (response(b"x", auth=False), "INVALID_TPM_RESPONSE_TAG"),
        (response(b"x", auth=True)[:-5] + bytes(5), "INVALID_PASSWORD_AUTH_RESPONSE"),
        (response(b"x", auth=True)[:-5] + bridge._PW_RESPONSE * 2, "TPM_RESPONSE_SIZE_MISMATCH"),
    ],
)
def test_certify_response_framing_and_authorization_rejected(
    collection: TestOnlyCollection, raw: bytes, error: str
) -> None:
    collection.tbs.overrides[bridge.TPM_CC_CERTIFY_CREATION] = raw
    with pytest.raises(bridge.WindowsCNGCustodyBridgeError, match=error):
        collection.collect()
    assert len(collection.tbs.commands) == 3
    assert ("free", 33) in collection.ncrypt.calls


def test_wrong_ak_signature_rejected(collection: TestOnlyCollection) -> None:
    collection.tbs.signature_override = b"\x00\x18\x00\x0b" + two_b(b"\x01") + two_b(b"\x01")
    with pytest.raises(bridge.WindowsCNGCustodyBridgeError, match="SIGNATURE_REJECTED"):
        collection.collect()


@pytest.mark.parametrize(
    "signature",
    [
        b"",
        b"\x00\x10",
        b"\x00\x18\x00\x0b" + two_b(b"") + two_b(b"1"),
        b"\x00\x18\x00\x0b" + two_b(b"1" * 33) + two_b(b"1"),
    ],
)
def test_malformed_signatures_rejected(collection: TestOnlyCollection, signature: bytes) -> None:
    collection.tbs.signature_override = signature
    with pytest.raises(bridge.WindowsCNGCustodyBridgeError, match="SIGNATURE"):
        collection.collect()


def test_certify_not_certify_creation_rejected(collection: TestOnlyCollection) -> None:
    collection.tbs.attest_override = struct.pack(">IH", 0xFF544347, 0x8017) + bytes(200)
    with pytest.raises(ProductionTPMCustodyError, match="CREATION"):
        collection.collect()


@pytest.mark.parametrize("field", ["signer", "qualifier", "name", "hash"])
def test_signed_attestation_binding_mismatch_rejected(
    collection: TestOnlyCollection, field: str
) -> None:
    values = {
        "signer": collection.tbs.qualified_name,
        "qualifier": pre_enrollment_custody_qualifying_data(collection.request),
        "name": name_of(collection.tbs.subject),
        "hash": b"c" * 32,
    }
    values[field] = b"\x00\x0b" + b"z" * 32 if field in {"signer", "name"} else b"z" * 32
    collection.tbs.attest_override = (
        struct.pack(">IH", 0xFF544347, 0x801A)
        + two_b(values["signer"])
        + two_b(values["qualifier"])
        + bytes(16)
        + b"\x01"
        + bytes(8)
        + two_b(values["name"])
        + two_b(values["hash"])
    )
    with pytest.raises(bridge.WindowsCNGCustodyBridgeError, match="BINDING_MISMATCH"):
        collection.collect()
    assert ("free", 33) in collection.ncrypt.calls


@pytest.mark.parametrize("name", ["", "not/a/key", cng.KEY_NAME])
def test_invalid_or_prekey_ak_identifier_rejected(
    collection: TestOnlyCollection, name: str
) -> None:
    with pytest.raises(bridge.WindowsCNGCustodyBridgeError, match="INVALID_RETAINED_AK_KEY_NAME"):
        bridge.certify_pre_enrollment_key_creation(
            collection.key,
            collection.request,
            ak_key_name=name,
            target_tpm_projection=collection.projection,
        )
    assert not collection.tbs.commands
    assert not any(call[0] == "open_ak" for call in collection.ncrypt.calls)


def test_changed_request_public_key_rejected(collection: TestOnlyCollection) -> None:
    changed = request_for(sec1(ec.generate_private_key(ec.SECP256R1()))).document
    retained = collection.request.document
    for field in (
        "pre_enrollment_public_key_canonical_bytes",
        "pre_enrollment_public_key_fingerprint_sha256",
    ):
        retained[field] = changed[field]
    collection.request = PreEnrollmentRequestV1.from_mapping(retained)
    with pytest.raises(bridge.WindowsCNGCustodyBridgeError, match="PRE_ENROLLMENT_KEY_MISMATCH"):
        collection.collect()
    assert not collection.tbs.commands


def test_missing_retained_ak_is_open_only(collection: TestOnlyCollection) -> None:
    def missing(*_arguments: object) -> int:
        return cng.NTE_BAD_KEYSET

    collection.ncrypt.NCryptOpenKey.function = missing
    with pytest.raises(cng.WindowsCNGPreEnrollmentError, match="OPEN_RETAINED_AK"):
        collection.collect()
    assert not collection.tbs.commands
    assert not any(
        call[0] in {"create", "sign", "set", "delete"} for call in collection.ncrypt.calls
    )
    assert ("free", 33) not in collection.ncrypt.calls


def test_ak_must_have_distinct_borrowed_object_handle(collection: TestOnlyCollection) -> None:
    collection.ncrypt.properties[(33, "PCP_PLATFORMHANDLE")] = struct.pack("<I", 0x80000001)
    with pytest.raises(bridge.WindowsCNGCustodyBridgeError, match="DISTINCT_RETAINED_AK_REQUIRED"):
        collection.collect()
    assert not collection.tbs.commands
    assert ("free", 33) in collection.ncrypt.calls


def test_native_tbs_cannot_be_substituted(
    collection: TestOnlyCollection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bridge, "_load_tbs_native", lambda: object())
    with pytest.raises(
        bridge.WindowsCNGCustodyBridgeError, match="EXACT_NATIVE_TBS_BOUNDARY_REQUIRED"
    ):
        collection.collect()
    assert not collection.tbs.commands


@pytest.mark.parametrize("version", [1, 0])
def test_tpm20_required_before_any_tpm_command(
    collection: TestOnlyCollection, version: int
) -> None:
    collection.tbs.device_version = version
    with pytest.raises(bridge.WindowsCNGCustodyBridgeError, match="TPM20_DEVICE_REQUIRED"):
        collection.collect()
    assert not collection.tbs.commands


def test_secure_tbs_loader_and_fixed_abi(monkeypatch: pytest.MonkeyPatch) -> None:
    dll = TestOnlyTBSDLL(b"", b"", ec.generate_private_key(ec.SECP256R1()))
    calls: list[tuple] = []

    def load(name: str, **arguments: object) -> TestOnlyTBSDLL:
        calls.append((name, arguments))
        return dll

    monkeypatch.setattr(bridge.sys, "platform", "win32")
    monkeypatch.setattr(bridge.ctypes, "WinDLL", load, raising=False)
    native = bridge._TBSCustodyAPI()
    assert calls == [("tbs.dll", {"use_last_error": True, "winmode": 0x800})]
    assert native.dll.Tbsip_Submit_Command.argtypes[0] is ctypes.c_void_p
    assert native.dll.Tbsip_Submit_Command.argtypes[-1] is ctypes.POINTER(ctypes.c_uint32)
    assert native.dll.Tbsip_Submit_Command.restype is ctypes.c_uint32
    assert not hasattr(native.dll, "Tbsip_Context_Close")
    assert not hasattr(native.dll, "Tbsi_Context_Create")


def test_non_windows_native_loader_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge.sys, "platform", "linux")
    with pytest.raises(bridge.WindowsCNGCustodyBridgeError, match="WINDOWS_REQUIRED"):
        bridge._load_tbs_native()
