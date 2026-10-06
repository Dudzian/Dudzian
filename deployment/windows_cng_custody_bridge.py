"""Collect CertifyCreation for the exact reopened production PCP identity.

The PCP owns its TBS context and virtual TPM handles: this bridge never creates,
closes or flushes them. It opens an existing machine AK through the same provider
and returns public evidence, never an authenticated capability. Production
acceptance remains the independent custody verifier's responsibility.

Supported creation-property codecs are explicitly bounded: CREATIONHASH is
exactly a raw SHA-256 digest or a marshaled TPM2B_DIGEST(size=32); CREATIONTICKET
is exactly a marshaled TPMT_TK_CREATION. Native C structures and other formats
are rejected. SDK property constants do not establish availability on a reopened
persistent ECDSA key. That Windows/physical-TPM qualification remains NOT_RUN.
"""

from __future__ import annotations

import ctypes
import hashlib
import re
import struct
import sys
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

from bot_core.licensing.device_enrollment import TPMPublicProjectionV1
from bot_core.licensing.pre_enrollment import PreEnrollmentRequestV1, validate_public_key
from bot_core.licensing.production_tpm_custody import (
    ProductionPreEnrollmentKeyCustodyEvidenceV1,
    make_pre_enrollment_custody_evidence,
    parse_production_creation_attestation,
    parse_production_ecc_public,
    pre_enrollment_custody_qualifying_data,
)
from deployment import windows_cng_pre_enrollment as cng

TPM_CC_READ_PUBLIC = 0x00000173
TPM_CC_CERTIFY_CREATION = 0x0000014A
TPM_ST_NO_SESSIONS = 0x8001
TPM_ST_SESSIONS = 0x8002
TPM_ST_CREATION = 0x8021
TPM_RS_PW = 0x40000009
MAX_RESPONSE = 4096
_PW_RESPONSE = b"\x00\x00\x01\x00\x00"


class WindowsCNGCustodyBridgeError(RuntimeError):
    """A native resource, codec, profile or exact binding failed closed."""


def _u16(value: int) -> bytes:
    return struct.pack(">H", value)


def _u32(value: int) -> bytes:
    return struct.pack(">I", value)


def _tpm2b(value: bytes) -> bytes:
    if len(value) > 0xFFFF:
        raise WindowsCNGCustodyBridgeError("TPM2B_TOO_LARGE")
    return _u16(len(value)) + value


def _read_2b(raw: bytes, offset: int) -> tuple[bytes, int]:
    if offset + 2 > len(raw):
        raise WindowsCNGCustodyBridgeError("MALFORMED_TPM2B")
    size = int.from_bytes(raw[offset : offset + 2], "big")
    end = offset + 2 + size
    if end > len(raw):
        raise WindowsCNGCustodyBridgeError("MALFORMED_TPM2B")
    return raw[offset + 2 : end], end


def _packet(code: int, handles: tuple[int, ...], parameters: bytes, *, auth: bool) -> bytes:
    body = b"".join(_u32(handle) for handle in handles)
    if auth:
        # CertifyCreation authorizes signHandle only. No object authorization.
        password = _u32(TPM_RS_PW) + b"\x00\x00\x00\x00\x00"
        body += _u32(len(password)) + password
    body += parameters
    tag = TPM_ST_SESSIONS if auth else TPM_ST_NO_SESSIONS
    return _u16(tag) + _u32(10 + len(body)) + _u32(code) + body


def _response(response: bytes, *, auth: bool) -> bytes:
    if not 10 <= len(response) <= MAX_RESPONSE:
        raise WindowsCNGCustodyBridgeError("MALFORMED_TPM_RESPONSE")
    tag, size, status = struct.unpack_from(">HII", response)
    if size != len(response):
        raise WindowsCNGCustodyBridgeError("TPM_RESPONSE_SIZE_MISMATCH")
    if status:
        raise WindowsCNGCustodyBridgeError(f"TPM_COMMAND_REJECTED:0x{status:08X}")
    if tag != (TPM_ST_SESSIONS if auth else TPM_ST_NO_SESSIONS):
        raise WindowsCNGCustodyBridgeError("INVALID_TPM_RESPONSE_TAG")
    if not auth:
        return response[10:]
    if len(response) < 14:
        raise WindowsCNGCustodyBridgeError("MISSING_TPM_PARAMETER_SIZE")
    parameter_size = int.from_bytes(response[10:14], "big")
    end = 14 + parameter_size
    if end > len(response) or response[end:] != _PW_RESPONSE:
        raise WindowsCNGCustodyBridgeError("INVALID_PASSWORD_AUTH_RESPONSE")
    return response[14:end]


class _TBSCustodyAPI:
    """Fixed, borrowed-context ABI with no context create/close entrypoint."""

    dll: Any

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise WindowsCNGCustodyBridgeError("WINDOWS_REQUIRED")
        self.dll = ctypes.WinDLL("tbs.dll", use_last_error=True, winmode=0x800)
        self.dll.Tbsip_Submit_Command.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.dll.Tbsip_Submit_Command.restype = ctypes.c_uint32
        self.dll.Tbsi_GetDeviceInfo.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        self.dll.Tbsi_GetDeviceInfo.restype = ctypes.c_uint32

    def require_tpm20(self) -> None:
        information = (ctypes.c_uint32 * 4)()
        status = self.dll.Tbsi_GetDeviceInfo(ctypes.sizeof(information), information)
        if status or information[0] != 1 or information[1] != 2:
            raise WindowsCNGCustodyBridgeError("TPM20_DEVICE_REQUIRED")

    def submit(self, context: int, request: bytes, *, auth: bool) -> bytes:
        source = (ctypes.c_ubyte * len(request)).from_buffer_copy(request)
        output = (ctypes.c_ubyte * MAX_RESPONSE)()
        size = ctypes.c_uint32(MAX_RESPONSE)
        status = self.dll.Tbsip_Submit_Command(
            context, 0, 200, source, len(request), output, ctypes.byref(size)
        )
        if status:
            raise WindowsCNGCustodyBridgeError(f"TBS_SUBMIT_FAILED:0x{status:08X}")
        if not 10 <= size.value <= MAX_RESPONSE:
            raise WindowsCNGCustodyBridgeError("INVALID_TBS_RESPONSE_SIZE")
        return _response(bytes(output[: size.value]), auth=auth)


def _load_tbs_native() -> _TBSCustodyAPI:
    return _TBSCustodyAPI()


def _platform_handle(native: cng._NCryptAPI, handle: int, *, provider: bool) -> int:
    raw = native.property(handle, "PCP_PLATFORMHANDLE")
    width = ctypes.sizeof(ctypes.c_void_p) if provider else 4
    if len(raw) != width:
        raise WindowsCNGCustodyBridgeError("INVALID_PCP_PLATFORM_HANDLE")
    value = int.from_bytes(raw, "little")
    if not value or (not provider and not 0x80000000 <= value <= 0x80FFFFFF):
        raise WindowsCNGCustodyBridgeError("INVALID_PCP_PLATFORM_HANDLE")
    return value


def _creation_hash(raw: bytes) -> bytes:
    if len(raw) == 32:
        return raw
    if len(raw) == 34 and raw[:2] == b"\x00\x20":
        return raw[2:]
    raise WindowsCNGCustodyBridgeError("UNSUPPORTED_PCP_CREATION_HASH_CODEC")


def _creation_ticket(raw: bytes) -> bytes:
    if len(raw) < 8:
        raise WindowsCNGCustodyBridgeError("UNSUPPORTED_PCP_CREATION_TICKET_CODEC")
    tag, hierarchy = struct.unpack_from(">HI", raw)
    ticket_digest, end = _read_2b(raw, 6)
    if (
        tag != TPM_ST_CREATION
        or hierarchy not in {0x40000001, 0x4000000B, 0x4000000C}
        or len(ticket_digest) not in {32, 64}
        or end != len(raw)
    ):
        raise WindowsCNGCustodyBridgeError("UNSUPPORTED_PCP_CREATION_TICKET_CODEC")
    return raw


def _read_public(tbs: _TBSCustodyAPI, context: int, handle: int) -> tuple[bytes, bytes, bytes]:
    parameters = tbs.submit(
        context, _packet(TPM_CC_READ_PUBLIC, (handle,), b"", auth=False), auth=False
    )
    public, offset = _read_2b(parameters, 0)
    name, offset = _read_2b(parameters, offset)
    qualified_name, offset = _read_2b(parameters, offset)
    if offset != len(parameters) or len(qualified_name) != 34 or qualified_name[:2] != b"\x00\x0b":
        raise WindowsCNGCustodyBridgeError("INVALID_READ_PUBLIC_RESPONSE")
    return public, name, qualified_name


def _open_retained_ak(native: cng._NCryptAPI, provider: int, name: str) -> int:
    handle = cng._HANDLE()
    cng._check(
        native.dll.NCryptOpenKey(provider, ctypes.byref(handle), name, 0, cng.MACHINE_KEY),
        "OPEN_RETAINED_AK",
    )
    if not handle.value:
        raise WindowsCNGCustodyBridgeError("EMPTY_RETAINED_AK_HANDLE")
    return int(handle.value)


def _require_ak_profile(native: cng._NCryptAPI, ak: int, name: str) -> None:
    if (
        native.text(ak, "Name") != name
        or native.number(ak, "Key Type") != cng.MACHINE_KEY
        or native.number(ak, "PCP_KEY_USAGE_POLICY") != 8
        or native.number(ak, "Export Policy") != 0
        or native.export_allowed(ak)
        or native.property(ak, "PCP_PASSWORD_REQUIRED") != b"\x00"
    ):
        raise WindowsCNGCustodyBridgeError("RETAINED_AK_PROFILE_UNSUPPORTED")


def _signature_der(raw: bytes) -> bytes:
    if raw[:4] != b"\x00\x18\x00\x0b":
        raise WindowsCNGCustodyBridgeError("INVALID_CERTIFY_CREATION_SIGNATURE")
    r, offset = _read_2b(raw, 4)
    s, offset = _read_2b(raw, offset)
    if (
        offset != len(raw)
        or not 1 <= len(r) <= 32
        or not 1 <= len(s) <= 32
        or not 0 < int.from_bytes(r, "big") < cng.P256_ORDER
        or not 0 < int.from_bytes(s, "big") < cng.P256_ORDER
    ):
        raise WindowsCNGCustodyBridgeError("INVALID_CERTIFY_CREATION_SIGNATURE")
    return bytes(utils.encode_dss_signature(int.from_bytes(r, "big"), int.from_bytes(s, "big")))


def certify_pre_enrollment_key_creation(
    key: cng.WindowsCNGPreEnrollmentKey,
    request: PreEnrollmentRequestV1,
    *,
    ak_key_name: str,
    target_tpm_projection: TPMPublicProjectionV1,
) -> ProductionPreEnrollmentKeyCustodyEvidenceV1:
    """Collect raw evidence without creating an AK or issuing authority.

    The caller retains the enrollment AK's existing CNG machine key identifier.
    An unavailable key, password/policy profile or creation property is terminal
    for this collection attempt; reconciliation and key provisioning are separate.
    """
    # Registry provenance and durable continuity are checked by the key factory.
    key = cng.require_verified_production_cng_key(key)
    if (
        type(request) is not PreEnrollmentRequestV1
        or type(target_tpm_projection) is not TPMPublicProjectionV1
    ):
        raise WindowsCNGCustodyBridgeError("EXACT_CANONICAL_BINDINGS_REQUIRED")
    if (
        not isinstance(ak_key_name, str)
        or not re.fullmatch(r"[A-Za-z0-9._\\:-]{1,256}", ak_key_name)
        or ak_key_name == cng.KEY_NAME
    ):
        raise WindowsCNGCustodyBridgeError("INVALID_RETAINED_AK_KEY_NAME")
    request = PreEnrollmentRequestV1.from_canonical_bytes(request.canonical_bytes)
    projection = TPMPublicProjectionV1.from_canonical_bytes(target_tpm_projection.canonical_bytes)
    document, target = request.document, projection.document
    expected = {
        "verified_tpm_public_projection_id": projection.evidence_reference,
        "ek_public_digest": target["ek"]["public_digest"],
        "ak_public_digest": target["ak"]["public_digest"],
    }
    if any(document[field] != value for field, value in expected.items()):
        raise WindowsCNGCustodyBridgeError("REQUEST_TPM_PROJECTION_MISMATCH")
    public, unique = key._qualify()
    if (
        public != key.public_key_bytes
        or unique != key._unique_name
        or document["pre_enrollment_public_key_canonical_bytes"] != public.hex()
    ):
        raise WindowsCNGCustodyBridgeError("EXACT_PRE_ENROLLMENT_KEY_MISMATCH")
    native = key._native
    context = _platform_handle(native, key._provider, provider=True)
    subject_handle = _platform_handle(native, key._key, provider=False)
    try:
        creation_hash = _creation_hash(native.property(key._key, "PCP_KEY_CREATIONHASH"))
        ticket = _creation_ticket(native.property(key._key, "PCP_KEY_CREATIONTICKET"))
    except cng.WindowsCNGPreEnrollmentError as exc:
        raise WindowsCNGCustodyBridgeError("PCP_CREATION_EVIDENCE_UNAVAILABLE") from exc
    tbs = _load_tbs_native()
    if type(tbs) is not _TBSCustodyAPI:
        raise WindowsCNGCustodyBridgeError("EXACT_NATIVE_TBS_BOUNDARY_REQUIRED")
    tbs.require_tpm20()
    ak = _open_retained_ak(native, key._provider, ak_key_name)
    try:
        _require_ak_profile(native, ak, ak_key_name)
        ak_handle = _platform_handle(native, ak, provider=False)
        if ak_handle == subject_handle:
            raise WindowsCNGCustodyBridgeError("DISTINCT_RETAINED_AK_REQUIRED")
        ak_raw, ak_name, ak_qualified_name = _read_public(tbs, context, ak_handle)
        ak_public = parse_production_ecc_public(ak_raw, role="ak")
        if (
            ak_public.auth_policy
            or not ak_public.attributes & 0x40
            or ak_name != ak_public.name
            or ak_raw.hex() != target["ak"]["public_area"]["hex"]
            or ak_name.hex() != target["ak"]["name"]
        ):
            raise WindowsCNGCustodyBridgeError("RETAINED_AK_IDENTITY_OR_AUTH_MISMATCH")
        subject_raw, subject_name, _qualified_name = _read_public(tbs, context, subject_handle)
        subject = parse_production_ecc_public(subject_raw, role="pre_enrollment")
        if subject.sec1 != public or subject.name != subject_name:
            raise WindowsCNGCustodyBridgeError("PCP_TPM_PUBLIC_IDENTITY_MISMATCH")
        qualifying = pre_enrollment_custody_qualifying_data(request)
        parameters = _tpm2b(qualifying) + _tpm2b(creation_hash) + b"\x00\x18\x00\x0b" + ticket
        result = tbs.submit(
            context,
            _packet(TPM_CC_CERTIFY_CREATION, (ak_handle, subject_handle), parameters, auth=True),
            auth=True,
        )
        attest, offset = _read_2b(result, 0)
        signature = result[offset:]
        parsed = parse_production_creation_attestation(attest)
        if (
            parsed.qualified_signer != ak_qualified_name
            or parsed.extra_data != qualifying
            or parsed.object_name != subject_name
            or parsed.creation_hash != creation_hash
        ):
            raise WindowsCNGCustodyBridgeError("CERTIFY_CREATION_BINDING_MISMATCH")
        try:
            validate_public_key(ak_public.sec1).verify(
                _signature_der(signature),
                hashlib.sha256(attest).digest(),
                ec.ECDSA(utils.Prehashed(hashes.SHA256())),
            )
        except InvalidSignature as exc:
            raise WindowsCNGCustodyBridgeError("CERTIFY_CREATION_SIGNATURE_REJECTED") from exc
        return make_pre_enrollment_custody_evidence(
            request=request,
            target_tpm_projection=projection,
            subject_tpmt_public=subject_raw,
            subject_name=subject_name,
            creation_hash=creation_hash,
            attest=attest,
            signature=signature,
        )
    finally:
        # Only this NCrypt reference is owned here. PCP releases its TPM resource.
        native.free(ak)
