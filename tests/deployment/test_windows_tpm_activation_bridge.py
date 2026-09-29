from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

from bot_core.licensing.tpm_attestation import proof_of_possession_digest
from deployment.windows_tpm_activation_bridge import (
    PhysicalWindowsTPMEnrollmentSubstrate,
    PrimaryPublic,
    TPM_RH_ENDORSEMENT,
    TPM_RH_OWNER,
    TPM_ST_NO_SESSIONS,
    TPM_ST_SESSIONS,
    parse_tpm_response,
    physical_report,
)
from deployment.windows_tpm_substrate_probe import ProbeError, read_tpm2b, tpm2b, u16, u32

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/windows_stage9_policy_vector_v1.json"


def response(tag: int, body: bytes, *, rc: int = 0) -> bytes:
    return u16(tag) + u32(10 + len(body)) + u32(rc) + body


def auth_response() -> bytes:
    return tpm2b(b"") + b"\x00" + tpm2b(b"")


def public_area() -> bytes:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return bytes.fromhex(fixture["enrollment_policy_material"]["k_psa"]["public_hex"])


class FakePhysicalSubstrate(PhysicalWindowsTPMEnrollmentSubstrate):
    def __init__(self, public: bytes) -> None:
        self.public = public
        self.calls: list[int] = []

    def create_primary(self, hierarchy: int, template: bytes) -> PrimaryPublic:
        self.calls.append(hierarchy)
        selected = self.public if len(self.calls) == 1 else template
        name = b"\x00\x0b" + hashlib.sha256(selected).digest()
        return PrimaryPublic(
            0x80000000 + len(self.calls),
            selected,
            name,
            b"\x77" * 32,
            u16(0x8021) + u32(TPM_RH_OWNER) + tpm2b(b"\x88" * 32),
        )


def create_primary_parameters(public: bytes) -> bytes:
    name = b"\x00\x0b" + hashlib.sha256(public).digest()
    ticket = u16(0x8021) + u32(TPM_RH_OWNER) + tpm2b(b"\x88" * 32)
    return tpm2b(public) + tpm2b(b"creation-data") + tpm2b(b"\x77" * 32) + ticket + tpm2b(name)


def test_physical_projection_uses_owner_kpsa_and_ak_plus_endorsement_ek():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    public = bytes.fromhex(fixture["enrollment_policy_material"]["k_psa"]["public_hex"])
    substrate = FakePhysicalSubstrate(public)
    evidence = substrate.collect().document
    assert substrate.calls == [TPM_RH_OWNER, TPM_RH_ENDORSEMENT, TPM_RH_OWNER]
    assert evidence["k_psa"]["name"] == fixture["policy_vector"]["k_psa_name"]
    assert evidence["source"]["substrate"] == "Windows-TBS"
    assert set(evidence) == {
        "schema",
        "version",
        "evidence_profile",
        "k_psa",
        "ek",
        "ak",
        "tpm",
        "source",
        "evidence_id",
    }


def test_adapter_contains_no_persistent_or_nv_lifecycle_commands():
    source = Path("deployment/windows_tpm_activation_bridge.py").read_text(encoding="utf-8")
    for forbidden in (
        "TPM2_Clear",
        "ClearControl",
        "TPM_RH_PLATFORM",
        "TPMA_NV_PLATFORMCREATE",
        "EvictControl",
    ):
        assert forbidden not in source
    assert (
        "NvProbe.flush" not in source
    )  # cleanup is invoked via the reused object, not reimplemented


def test_create_primary_sessions_response_separates_handle_and_parameter_size():
    expected_public = public_area()
    parameters = create_primary_parameters(expected_public)
    raw = response(
        TPM_ST_SESSIONS,
        u32(0x80000000) + u32(len(parameters)) + parameters + auth_response(),
    )
    parsed = parse_tpm_response(raw, response_handle_count=1)
    returned_public, _ = read_tpm2b(parsed.parameters)
    assert parsed.handles == (0x80000000,)
    assert returned_public == expected_public


def test_create_primary_consumes_handle_from_response_handle_area():
    expected_public = public_area()
    parameters = create_primary_parameters(expected_public)
    raw = response(
        TPM_ST_SESSIONS,
        u32(0x80000000) + u32(len(parameters)) + parameters + auth_response(),
    )

    class Transport:
        def submit(self, code: int, request: bytes) -> bytes:
            assert code != 0 and request
            return raw

    substrate = object.__new__(PhysicalWindowsTPMEnrollmentSubstrate)
    substrate.transport = Transport()
    substrate.handles = []
    substrate.read_public = lambda handle: (
        expected_public,
        b"\x00\x0b" + hashlib.sha256(expected_public).digest(),
    )
    created = substrate.create_primary(TPM_RH_OWNER, expected_public)
    assert created.handle == 0x80000000
    assert created.public == expected_public
    assert created.creation_hash == b"\x77" * 32
    assert created.creation_ticket == u16(0x8021) + u32(TPM_RH_OWNER) + tpm2b(b"\x88" * 32)
    assert substrate.handles == [0x80000000]


@pytest.mark.parametrize(
    "parameters",
    [
        bytes.fromhex("0018000b") + tpm2b(b"\x01" * 32) + tpm2b(b"\x02" * 32),
        tpm2b(b"attest") + bytes.fromhex("0018000b") + tpm2b(b"\x01" * 32) + tpm2b(b"\x02" * 32),
        tpm2b(b"activated-credential"),
    ],
)
def test_new_tpm_commands_use_zero_handle_sessions_response_framing(parameters: bytes):
    raw = response(TPM_ST_SESSIONS, u32(len(parameters)) + parameters + auth_response())
    parsed = parse_tpm_response(raw, response_handle_count=0)
    assert parsed.handles == ()
    assert parsed.parameters == parameters


def test_no_sessions_response_with_handle_has_no_parameter_size_field():
    parameters = tpm2b(public_area())
    parsed = parse_tpm_response(
        response(TPM_ST_NO_SESSIONS, u32(0x80000000) + parameters),
        response_handle_count=1,
    )
    assert parsed.handles == (0x80000000,)
    assert parsed.parameters == parameters


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (response(TPM_ST_SESSIONS, b"\x80\x00"), "TRUNCATED_RESPONSE_HANDLE_AREA"),
        (
            response(TPM_ST_SESSIONS, u32(0x80000000) + u32(99) + b"short"),
            "PARAMETER_SIZE_OUT_OF_BOUNDS",
        ),
        (response(TPM_ST_SESSIONS, u32(0x80000000) + b"\x00\x01"), "MISSING_PARAMETER_SIZE"),
        (response(0x8123, b""), "INVALID_TPM_RESPONSE_TAG"),
        (response(TPM_ST_NO_SESSIONS, b"", rc=0x101), "TPM_COMMAND_REJECTED"),
        (
            response(TPM_ST_SESSIONS, u32(1) + tpm2b(public_area()) + auth_response()),
            "PARAMETER_SIZE_OUT_OF_BOUNDS",
        ),
        (
            response(
                TPM_ST_SESSIONS,
                u32(0x80000000) + u32(0x80000001) + u32(0) + auth_response(),
            ),
            "PARAMETER_SIZE_OUT_OF_BOUNDS",
        ),
    ],
)
def test_handle_response_framing_rejects_malformed_boundaries(raw: bytes, reason: str):
    with pytest.raises(ProbeError, match=reason):
        parse_tpm_response(raw, response_handle_count=1)


def test_sessions_response_rejects_malformed_authorization_boundary():
    raw = response(TPM_ST_SESSIONS, u32(0x80000000) + u32(0) + b"\x00")
    with pytest.raises(ProbeError, match="MALFORMED_AUTHORIZATION_AREA"):
        parse_tpm_response(raw, response_handle_count=1)


def test_preflight_report_never_claims_unimplemented_attestation_pass():
    substrate = FakePhysicalSubstrate(public_area())
    report = physical_report(substrate.collect())
    assert report["public_tpm_projection"] == "IMPLEMENTED"
    assert report["k_psa_tpm_sign"] == "NOT_RUN"
    assert report["activate_credential"] == "NOT_RUN"
    assert report["certify_creation"] == "NOT_RUN"
    assert report["issuer_exchange"] == "NOT_RUN"
    assert report["activation_request"] == "BLOCKED_ATTESTATION_REQUIRED"


def test_cli_has_explicit_preflight_physical_and_negative_flows():
    source = Path("scripts/cryptohunter_activation_request.py").read_text(encoding="utf-8")
    assert 'print("ACTIVATION REQUEST = BLOCKED", file=sys.stderr)' in source
    assert "return 3" in source
    assert 'args.command == "physical-test"' in source
    assert 'args.command == "negative-test"' in source


class FakePolicyObjects:
    def __init__(self, digest: bytes) -> None:
        self.digest = digest
        self.flushed: list[int] = []

    def start_policy(self):
        return 0x03000000, b"caller", b"tpm"

    def policy_command_code(self, session, code):
        assert session == 0x03000000 and code != 0

    def policy_digest(self, session):
        return self.digest

    def flush(self, handle):
        self.flushed.append(handle)


class QueueTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def submit(self, code, request):
        self.requests.append((code, request))
        return self.responses.pop(0)


def session_response(parameters: bytes) -> bytes:
    return response(TPM_ST_SESSIONS, u32(len(parameters)) + parameters + auth_response())


def start_policy_response() -> bytes:
    return response(TPM_ST_NO_SESSIONS, u32(0x03000000) + tpm2b(b"\x55" * 32))


def test_kpsa_policy_sign_crosscheck_host_verify_and_session_cleanup():
    from bot_core.licensing.tpm_attestation import test_only_sign_policy_digest

    public = public_area()
    nonce = b"\x42" * 32
    der = ec.derive_private_key(1, ec.SECP256R1()).sign(
        proof_of_possession_digest(nonce), ec.ECDSA(utils.Prehashed(hashes.SHA256()))
    )
    r, s = utils.decode_dss_signature(der)
    signature = (
        bytes.fromhex("0018000b") + tpm2b(r.to_bytes(32, "big")) + tpm2b(s.to_bytes(32, "big"))
    )
    substrate = object.__new__(PhysicalWindowsTPMEnrollmentSubstrate)
    substrate.objects = FakePolicyObjects(test_only_sign_policy_digest())
    substrate.transport = QueueTransport([start_policy_response(), session_response(signature)])
    substrate.handles = []
    primary = PrimaryPublic(
        0x80000000, public, b"\x00\x0b" + hashlib.sha256(public).digest(), b"\x77" * 32, b"ticket"
    )
    assert substrate.sign_k_psa(primary, nonce) == der
    assert substrate.objects.flushed == [0x03000000]
    assert substrate.handles == []


def test_kpsa_policy_digest_mismatch_fails_and_still_flushes():
    substrate = object.__new__(PhysicalWindowsTPMEnrollmentSubstrate)
    substrate.objects = FakePolicyObjects(b"\x00" * 32)
    substrate.transport = QueueTransport([start_policy_response()])
    substrate.handles = []
    public = public_area()
    primary = PrimaryPublic(
        1, public, b"\x00\x0b" + hashlib.sha256(public).digest(), b"\x77" * 32, b"ticket"
    )
    with pytest.raises(ProbeError, match="TEST_ONLY_K_PSA_POLICY_DIGEST_MISMATCH"):
        substrate.sign_k_psa(primary, b"\x42" * 32)
    assert substrate.objects.flushed == [0x03000000]


def test_certify_creation_uses_returned_creation_hash_and_ticket():
    attest = b"attest"
    signature = bytes.fromhex("0018000b") + tpm2b(b"\x01" * 32) + tpm2b(b"\x02" * 32)
    transport = QueueTransport([session_response(tpm2b(attest) + signature)])
    substrate = object.__new__(PhysicalWindowsTPMEnrollmentSubstrate)
    substrate.transport = transport
    ak = PrimaryPublic(1, b"ak", b"ak-name", b"\x11" * 32, b"ak-ticket")
    kpsa = PrimaryPublic(2, b"psa", b"psa-name", b"\x22" * 32, b"creation-ticket")
    assert substrate.certify_creation(ak, kpsa, b"qualifying") == (attest, signature)
    assert b"creation-ticket" in transport.requests[0][1]
    assert b"\x22" * 32 in transport.requests[0][1]


def test_activate_credential_two_session_framing_and_cleanup():
    recovered = b"credential"
    transport = QueueTransport(
        [start_policy_response(), session_response(b""), session_response(tpm2b(recovered))]
    )
    substrate = object.__new__(PhysicalWindowsTPMEnrollmentSubstrate)
    substrate.transport = transport
    substrate.objects = FakePolicyObjects(b"")
    substrate.handles = []
    ak = PrimaryPublic(1, b"ak", b"ak-name", b"h", b"t")
    ek = PrimaryPublic(2, b"ek", b"ek-name", b"h", b"t")
    assert substrate.activate_credential(ak, ek, b"blob", b"secret") == recovered
    assert b"blob" in transport.requests[2][1] and b"secret" in transport.requests[2][1]
    assert substrate.objects.flushed == [0x03000000]
