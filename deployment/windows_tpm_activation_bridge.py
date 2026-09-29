"""TEST_ONLY physical TPM enrollment adapter built on the Stage-9 TBS substrate.

Objects are deterministic transient primaries.  No persistent handle or NV index is
allocated; every handle is flushed before TBS is closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import secrets
from typing import Any

from bot_core.licensing.device_enrollment import TPMPublicProjectionV1, make_evidence
from bot_core.licensing.tpm_attestation import (
    certify_qualifying_data,
    proof_of_possession_digest,
    test_only_sign_policy_digest,
    tpm_ecdsa_signature_to_der,
    verify_k_psa_public_signature,
)
from deployment.windows_tpm_substrate_probe import (
    NvProbe,
    ProbeError,
    TbsTransport,
    blank_evidence,
    command_packet,
    decode_tpm_rc,
    read_tpm2b,
    read_u16,
    read_u32,
    response_parameters,
    tpm2b,
    u16,
    u32,
)
from deployment.windows_stage9_root_of_trust_freeze import validate_tpmt_public

TPM_CC_CREATE_PRIMARY = 0x00000131
TPM_CC_SIGN = 0x0000015D
TPM_CC_CERTIFY_CREATION = 0x0000014A
TPM_CC_ACTIVATE_CREDENTIAL = 0x00000147
TPM_CC_POLICY_SECRET = 0x00000151
TPM_CC_START_AUTH_SESSION = 0x00000176
TPM_ST_NO_SESSIONS = 0x8001
TPM_ST_SESSIONS = 0x8002
TPM_RH_OWNER = 0x40000001
TPM_RH_ENDORSEMENT = 0x4000000B
TPM_RH_NULL = 0x40000007
TPM_RS_PW = 0x40000009
TPM_ALG_SHA256 = 0x000B
TPM_ALG_NULL = 0x0010
TPM_ALG_ECDSA = 0x0018
TPM_ALG_ECC = 0x0023
TPM_ALG_AES = 0x0006
TPM_ALG_CFB = 0x0043
TPM_ECC_NIST_P256 = 0x0003
TPM_ST_HASHCHECK = 0x8024
TPM_SE_POLICY = 0x01
K_PSA_ATTRIBUTES = 0x000400B2
AK_ATTRIBUTES = 0x00050072
EK_ATTRIBUTES = 0x000300B2
TEST_POLICY = test_only_sign_policy_digest()
EK_POLICY = bytes.fromhex("837197674484b3f81a90cc8d46a5d724fd52d76e06520b64f2a1da1b331469aa")


def _ecc_template(attributes: int, policy: bytes, *, signing: bool, decrypt: bool = False) -> bytes:
    scheme = u16(TPM_ALG_ECDSA) + u16(TPM_ALG_SHA256) if signing else u16(TPM_ALG_NULL)
    symmetric = u16(TPM_ALG_AES) + u16(128) + u16(TPM_ALG_CFB) if decrypt else u16(TPM_ALG_NULL)
    return (
        u16(TPM_ALG_ECC)
        + u16(TPM_ALG_SHA256)
        + u32(attributes)
        + tpm2b(policy)
        + symmetric
        + scheme
        + u16(TPM_ECC_NIST_P256)
        + u16(TPM_ALG_NULL)
        + tpm2b(b"")
        + tpm2b(b"")
    )


def _create_primary_request(hierarchy: int, public: bytes) -> bytes:
    sensitive_create = tpm2b(b"") + tpm2b(b"")
    parameters = tpm2b(sensitive_create) + tpm2b(public) + tpm2b(b"") + u32(0)
    return command_packet(TPM_CC_CREATE_PRIMARY, (hierarchy,), parameters, auth=True)


def _session_command_packet(
    code: int,
    handles: tuple[int, ...],
    parameters: bytes,
    sessions: tuple[int, ...],
) -> bytes:
    authorization = b"".join(
        u32(session) + tpm2b(b"") + b"\x00" + tpm2b(b"") for session in sessions
    )
    body = b"".join(u32(handle) for handle in handles)
    body += u32(len(authorization)) + authorization + parameters
    return u16(TPM_ST_SESSIONS) + u32(10 + len(body)) + u32(code) + body


def _signature_extent(data: bytes, offset: int = 0) -> int:
    algorithm, offset = read_u16(data, offset)
    hash_algorithm, offset = read_u16(data, offset)
    if algorithm != TPM_ALG_ECDSA or hash_algorithm != TPM_ALG_SHA256:
        raise ValueError("unexpected TPMT_SIGNATURE algorithm")
    _r, offset = read_tpm2b(data, offset)
    _s, offset = read_tpm2b(data, offset)
    return offset


@dataclass(frozen=True)
class PrimaryPublic:
    handle: int
    public: bytes
    name: bytes
    creation_hash: bytes
    creation_ticket: bytes


@dataclass(frozen=True)
class ParsedTPMResponse:
    handles: tuple[int, ...]
    parameters: bytes


def _validate_authorization_area(data: bytes) -> None:
    """Validate one or more TPMS_AUTH_RESPONSE structures to the exact boundary."""
    if not data:
        raise ProbeError("MALFORMED_AUTHORIZATION_AREA", "authorization area is missing")
    offset = 0
    while offset < len(data):
        try:
            _nonce, offset = read_tpm2b(data, offset)
            if offset >= len(data):
                raise ValueError("missing sessionAttributes")
            offset += 1
            _hmac, offset = read_tpm2b(data, offset)
        except ValueError as exc:
            raise ProbeError("MALFORMED_AUTHORIZATION_AREA", str(exc)) from exc
    if offset != len(data):
        raise ProbeError("MALFORMED_AUTHORIZATION_AREA", "trailing authorization bytes")


def parse_tpm_response(response: bytes, *, response_handle_count: int) -> ParsedTPMResponse:
    """Parse TPM response framing without confusing response handles with parameters."""
    if response_handle_count < 0:
        raise ValueError("response_handle_count cannot be negative")
    try:
        tag, offset = read_u16(response)
        size, offset = read_u32(response, offset)
        rc, offset = read_u32(response, offset)
    except ValueError as exc:
        raise ProbeError("MALFORMED_TPM_RESPONSE", str(exc)) from exc
    if tag not in {TPM_ST_NO_SESSIONS, TPM_ST_SESSIONS}:
        raise ProbeError("INVALID_TPM_RESPONSE_TAG", f"tag=0x{tag:04X}")
    if size != len(response):
        raise ProbeError("TPM_RESPONSE_SIZE_MISMATCH", f"header={size}, actual={len(response)}")
    if rc:
        raise ProbeError("TPM_COMMAND_REJECTED", decode_tpm_rc(rc))

    handles = []
    try:
        for _ in range(response_handle_count):
            handle, offset = read_u32(response, offset)
            handles.append(handle)
    except ValueError as exc:
        raise ProbeError("TRUNCATED_RESPONSE_HANDLE_AREA", str(exc)) from exc

    if tag == TPM_ST_NO_SESSIONS:
        return ParsedTPMResponse(tuple(handles), response[offset:])

    try:
        parameter_size, offset = read_u32(response, offset)
    except ValueError as exc:
        raise ProbeError("MISSING_PARAMETER_SIZE", str(exc)) from exc
    parameter_end = offset + parameter_size
    if parameter_end > len(response):
        raise ProbeError(
            "PARAMETER_SIZE_OUT_OF_BOUNDS",
            f"parameterSize={parameter_size}, available={len(response) - offset}",
        )
    parameters = response[offset:parameter_end]
    _validate_authorization_area(response[parameter_end:])
    return ParsedTPMResponse(tuple(handles), parameters)


class PhysicalWindowsTPMEnrollmentSubstrate:
    """Minimal consumer of Stage-9 transport, response codec, ReadPublic and cleanup."""

    def __init__(self) -> None:
        self.log = blank_evidence("activation-bridge-v1", "TEST_ONLY")
        self.transport = TbsTransport(self.log)
        self.objects = NvProbe(self.transport)
        self.handles: list[int] = []

    def __enter__(self) -> "PhysicalWindowsTPMEnrollmentSubstrate":
        self.transport.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        failures = []
        for handle in reversed(self.handles):
            try:
                self.objects.flush(handle)
            except Exception as cleanup_error:  # cleanup must continue and then fail closed
                failures.append(str(cleanup_error))
        self.transport.close()
        if self.log["cleanup"].get("tbs_context") != "PASS":
            failures.append(str(self.log["cleanup"].get("tbs_context", "TBS_CLOSE_NOT_RECORDED")))
        if failures:
            raise ProbeError("TEST_ONLY_CLEANUP_FAILED", "; ".join(failures))

    def create_primary(self, hierarchy: int, template: bytes) -> PrimaryPublic:
        raw_response = self.transport.submit(
            TPM_CC_CREATE_PRIMARY, _create_primary_request(hierarchy, template)
        )
        parsed = parse_tpm_response(raw_response, response_handle_count=1)
        handle = parsed.handles[0]
        try:
            returned_public, offset = read_tpm2b(parsed.parameters)
            _creation_data, offset = read_tpm2b(parsed.parameters, offset)
            creation_hash, offset = read_tpm2b(parsed.parameters, offset)
            ticket_start = offset
            _ticket_tag, offset = read_u16(parsed.parameters, offset)
            _ticket_hierarchy, offset = read_u32(parsed.parameters, offset)
            _ticket_digest, offset = read_tpm2b(parsed.parameters, offset)
            creation_ticket = parsed.parameters[ticket_start:offset]
            returned_name, offset = read_tpm2b(parsed.parameters, offset)
            if offset != len(parsed.parameters):
                raise ValueError("trailing CreatePrimary parameters")
        except ValueError as exc:
            raise ProbeError("MALFORMED_CREATE_PRIMARY_PARAMETERS", str(exc)) from exc
        self.handles.append(handle)
        # ReadPublic is the independent TPM observation; never trust CreatePrimary alone.
        public, name = self.read_public(handle)
        if public != returned_public:
            raise ProbeError("TPM_PUBLIC_MISMATCH", "CreatePrimary and ReadPublic disagree")
        expected = u16(TPM_ALG_SHA256) + hashlib.sha256(public).digest()
        if name != expected:
            raise ProbeError(
                "K_PSA_TPM_NAME_VERIFICATION_FAILED", "TPM Name does not match TPMT_PUBLIC"
            )
        if returned_name != name:
            raise ProbeError(
                "CREATE_PRIMARY_NAME_MISMATCH", "CreatePrimary and ReadPublic disagree"
            )
        return PrimaryPublic(handle, public, name, creation_hash, creation_ticket)

    def read_public(self, handle: int) -> tuple[bytes, bytes]:
        # Stage-9 NvProbe owns the canonical command transport but its read_public is NV-specific.
        request = command_packet(0x00000173, (handle,), b"", auth=False)
        data = response_parameters(self.transport.submit(0x00000173, request))
        public, offset = read_tpm2b(data)
        name, offset = read_tpm2b(data, offset)
        _qualified_name, offset = read_tpm2b(data, offset)
        if offset != len(data):
            raise ProbeError("MALFORMED_READ_PUBLIC_RESPONSE", "trailing response bytes")
        return public, name

    def _flush_registered(self, handle: int) -> None:
        self.objects.flush(handle)
        self.handles.remove(handle)

    def _start_policy_session(self) -> int:
        nonce = secrets.token_bytes(32)
        parameters = (
            tpm2b(nonce)
            + tpm2b(b"")
            + bytes((TPM_SE_POLICY,))
            + u16(TPM_ALG_NULL)
            + u16(TPM_ALG_SHA256)
        )
        request = command_packet(
            TPM_CC_START_AUTH_SESSION,
            (TPM_RH_NULL, TPM_RH_NULL),
            parameters,
            auth=False,
        )
        parsed = parse_tpm_response(
            self.transport.submit(TPM_CC_START_AUTH_SESSION, request),
            response_handle_count=1,
        )
        _nonce_tpm, offset = read_tpm2b(parsed.parameters)
        if offset != len(parsed.parameters):
            raise ProbeError("MALFORMED_START_AUTH_SESSION_RESPONSE", "trailing bytes")
        session = parsed.handles[0]
        self.handles.append(session)
        return session

    def sign_k_psa(self, k_psa: PrimaryPublic, issuer_nonce: bytes) -> bytes:
        session = self._start_policy_session()
        try:
            self.objects.policy_command_code(session, TPM_CC_SIGN)
            if self.objects.policy_digest(session) != test_only_sign_policy_digest():
                raise ProbeError(
                    "TEST_ONLY_K_PSA_POLICY_DIGEST_MISMATCH",
                    "STOP — TEST_ONLY K_PSA POLICY DIGEST MISMATCH.",
                )
            digest = proof_of_possession_digest(issuer_nonce)
            scheme = u16(TPM_ALG_ECDSA) + u16(TPM_ALG_SHA256)
            validation = u16(TPM_ST_HASHCHECK) + u32(TPM_RH_NULL) + tpm2b(b"")
            request = _session_command_packet(
                TPM_CC_SIGN,
                (k_psa.handle,),
                tpm2b(digest) + scheme + validation,
                (session,),
            )
            parsed = parse_tpm_response(
                self.transport.submit(TPM_CC_SIGN, request), response_handle_count=0
            )
            end = _signature_extent(parsed.parameters)
            if end != len(parsed.parameters):
                raise ProbeError("MALFORMED_SIGN_RESPONSE", "trailing signature bytes")
            der = tpm_ecdsa_signature_to_der(parsed.parameters)
            verify_k_psa_public_signature(k_psa.public.hex(), issuer_nonce, der)
            return der
        finally:
            self._flush_registered(session)

    def certify_creation(
        self, ak: PrimaryPublic, k_psa: PrimaryPublic, qualifying_data: bytes
    ) -> tuple[bytes, bytes]:
        scheme = u16(TPM_ALG_ECDSA) + u16(TPM_ALG_SHA256)
        parameters = (
            tpm2b(qualifying_data) + tpm2b(k_psa.creation_hash) + scheme + k_psa.creation_ticket
        )
        request = command_packet(
            TPM_CC_CERTIFY_CREATION,
            (ak.handle, k_psa.handle),
            parameters,
            auth=True,
        )
        parsed = parse_tpm_response(
            self.transport.submit(TPM_CC_CERTIFY_CREATION, request), response_handle_count=0
        )
        attest, offset = read_tpm2b(parsed.parameters)
        signature_end = _signature_extent(parsed.parameters, offset)
        if signature_end != len(parsed.parameters):
            raise ProbeError("MALFORMED_CERTIFY_CREATION_RESPONSE", "trailing bytes")
        return attest, parsed.parameters[offset:signature_end]

    def _policy_secret_endorsement(self, session: int) -> None:
        parameters = tpm2b(b"") + tpm2b(b"") + tpm2b(b"") + u32(0)
        request = _session_command_packet(
            TPM_CC_POLICY_SECRET,
            (TPM_RH_ENDORSEMENT, session),
            parameters,
            (TPM_RS_PW,),
        )
        parse_tpm_response(
            self.transport.submit(TPM_CC_POLICY_SECRET, request), response_handle_count=0
        )

    def activate_credential(
        self, ak: PrimaryPublic, ek: PrimaryPublic, credential_blob: bytes, encrypted_secret: bytes
    ) -> bytes:
        session = self._start_policy_session()
        try:
            self._policy_secret_endorsement(session)
            request = _session_command_packet(
                TPM_CC_ACTIVATE_CREDENTIAL,
                (ak.handle, ek.handle),
                tpm2b(credential_blob) + tpm2b(encrypted_secret),
                (TPM_RS_PW, session),
            )
            parsed = parse_tpm_response(
                self.transport.submit(TPM_CC_ACTIVATE_CREDENTIAL, request),
                response_handle_count=0,
            )
            recovered, offset = read_tpm2b(parsed.parameters)
            if offset != len(parsed.parameters):
                raise ProbeError("MALFORMED_ACTIVATE_CREDENTIAL_RESPONSE", "trailing bytes")
            return recovered
        finally:
            self._flush_registered(session)

    def provision(
        self,
    ) -> tuple[TPMPublicProjectionV1, PrimaryPublic, PrimaryPublic, PrimaryPublic]:
        k_psa = self.create_primary(
            TPM_RH_OWNER, _ecc_template(K_PSA_ATTRIBUTES, TEST_POLICY, signing=True)
        )
        parsed = validate_tpmt_public(
            k_psa.public.hex(),
            {
                "object_attributes": "000400b2",
                "auth_policy_size": 32,
                "type": "TPM_ALG_ECC",
                "name_algorithm": "TPM_ALG_SHA256",
                "scheme": "TPM_ALG_ECDSA",
                "scheme_hash": "TPM_ALG_SHA256",
                "curve": "TPM_ECC_NIST_P256",
                "kdf": "TPM_ALG_NULL",
            },
        )
        if parsed.name != k_psa.name.hex():
            raise ProbeError(
                "K_PSA_TPM_NAME_VERIFICATION_FAILED", "canonical Stage-9 parser disagrees"
            )
        ek = self.create_primary(
            TPM_RH_ENDORSEMENT, _ecc_template(EK_ATTRIBUTES, EK_POLICY, signing=False, decrypt=True)
        )
        ak = self.create_primary(TPM_RH_OWNER, _ecc_template(AK_ATTRIBUTES, b"", signing=True))
        evidence = make_evidence(
            public_area_hex=k_psa.public.hex(),
            returned_name=k_psa.name.hex(),
            creation_hash=k_psa.creation_hash.hex(),
            ek_public_area_hex=ek.public.hex(),
            ek_name=ek.name.hex(),
            ak_public_area_hex=ak.public.hex(),
            ak_name=ak.name.hex(),
            algorithm_profile="Stage9.K_PSA.ECC_P256_SHA256.TEST_ONLY",
            evidence_profile="WindowsTPM2-TBS-Stage9-v1",
            substrate_profile="Stage9-TBS-deterministic-transient-primary-v1",
        )
        return evidence, k_psa, ek, ak

    def collect(self) -> TPMPublicProjectionV1:
        evidence, _k_psa, _ek, _ak = self.provision()
        return evidence

    def answer_challenge(
        self,
        *,
        issuer_nonce: bytes,
        activation_request_digest: bytes,
        credential_blob: bytes,
        encrypted_secret: bytes,
    ) -> dict[str, bytes]:
        evidence, k_psa, ek, ak = self.provision()
        recovered = self.activate_credential(ak, ek, credential_blob, encrypted_secret)
        pop_signature = self.sign_k_psa(k_psa, issuer_nonce)
        qualifying = certify_qualifying_data(issuer_nonce, activation_request_digest)
        attest, certify_signature = self.certify_creation(ak, k_psa, qualifying)
        return {
            "credential": recovered,
            "k_psa_pop_signature_der": pop_signature,
            "certify_creation_attest": attest,
            "certify_creation_signature": certify_signature,
            "public_projection_id": bytes.fromhex(evidence.evidence_reference),
        }


def load_or_create_installation_id(path: Path) -> str:
    """Create once using OS CSPRNG; it is public state, not a machine fingerprint."""
    if path.exists():
        value = path.read_text(encoding="ascii").strip()
        if len(value) != 64 or bytes.fromhex(value).hex() != value:
            raise ValueError("invalid installation_id state")
        return value
    import secrets

    path.parent.mkdir(parents=True, exist_ok=True)
    value = secrets.token_hex(32)
    path.write_text(value + "\n", encoding="ascii")
    return value


def physical_report(evidence: TPMPublicProjectionV1) -> dict[str, Any]:
    value = evidence.document
    return {
        "schema": "CryptoHunterPhysicalTPMActivationTestEvidenceV1",
        "physical_tpm": "PASS",
        "windows_tbs": "PASS",
        "k_psa": {
            "created_or_reused": "DETERMINISTIC_TRANSIENT_PRIMARY",
            "name": value["k_psa"]["name"],
            "public_area_digest": hashlib.sha256(
                bytes.fromhex(value["k_psa"]["public_area"]["hex"])
            ).hexdigest(),
            "name_recomputed": "PASS",
        },
        "ek_evidence": "PASS",
        "ek_public_binding": "PASS",
        "ek_manufacturer_certificate": value["ek"]["manufacturer_certificate"],
        "ak_evidence": "PASS",
        "public_tpm_projection": "IMPLEMENTED",
        "k_psa_tpm_sign": "NOT_RUN",
        "activate_credential": "NOT_RUN",
        "certify_creation": "NOT_RUN",
        "issuer_exchange": "NOT_RUN",
        "activation_request": "BLOCKED_ATTESTATION_REQUIRED",
        "evidence_reference_binding": "NOT_RUN",
        "bundle_verification": "NOT_RUN",
        "private_material_exported": "NO",
        "production_material_touched": "NO",
    }
