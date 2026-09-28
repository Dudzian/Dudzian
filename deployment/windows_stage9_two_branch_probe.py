"""Disposable physical probe for the two genuine Stage-9 policy branches.

All keys are ephemeral TEST_ONLY material. This module does not provide a
production authority and never changes MSI or provisioning composition.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils
from deployment.windows_tpm_substrate_probe import (
    NvProbe,
    ProbeError,
    TbsTransport,
    VerificationTicket,
    TPM_ALG_SHA256,
    TPM_CC_NV_DEFINE_SPACE,
    TPM_CC_NV_INCREMENT,
    TPM_CC_POLICY_COMMAND_CODE,
    TPM_CC_POLICY_OR,
    TPM_RH_OWNER,
    TPM_ST_SESSIONS,
    _choose_unused,
    command_packet,
    ecc_public_area,
    ecdsa_signature,
    qualify_windows,
    response_parameters,
    read_tpm2b,
    tpm2b,
    u16,
    u32,
)

TPM_CC_POLICY_CP_HASH = 0x0000016E
ATTRS = 0x02040018
REF_NORMAL = hashlib.sha256(b"CryptoHunter.Stage9.TwoBranchProbe.Normal.v1").digest()
REF_RECOVERY = hashlib.sha256(b"CryptoHunter.Stage9.TwoBranchProbe.Recovery.v1").digest()


def h(*parts: bytes) -> bytes:
    return hashlib.sha256(b"".join(parts)).digest()


def branch(name: bytes, ref: bytes) -> bytes:
    return h(h(bytes(32), u32(0x16A), name), ref)


def cp_hash(name: bytes) -> bytes:
    return h(u32(TPM_CC_NV_INCREMENT), name, name)


def policy_auth_packet(
    code: int, handles: tuple[int, ...], session: int, params: bytes = b""
) -> bytes:
    auth = u32(session) + tpm2b(b"") + b"\x01" + tpm2b(b"")
    body = b"".join(u32(x) for x in handles) + u32(len(auth)) + auth + params
    return u16(TPM_ST_SESSIONS) + u32(10 + len(body)) + u32(code) + body


def sign_authorize(
    nv: NvProbe,
    session: int,
    approved: bytes,
    ref: bytes,
    key: ec.EllipticCurvePrivateKey,
    handle: int,
    name: bytes,
) -> VerificationTicket:
    digest = h(approved, ref)
    der = key.sign(digest, ec.ECDSA(utils.Prehashed(hashes.SHA256())))
    r, s = utils.decode_dss_signature(der)
    signature = ecdsa_signature(r.to_bytes(32, "big"), s.to_bytes(32, "big"))
    ticket = nv.verify_signature(handle, digest, signature)
    nv.policy_authorize(session, approved, ref, name, ticket)
    return ticket


def define(nv: NvProbe, handle: int, auth_policy: bytes) -> None:
    public = u32(handle) + u16(TPM_ALG_SHA256) + u32(ATTRS) + tpm2b(auth_policy) + u16(8)
    params = tpm2b(b"") + tpm2b(public)
    response_parameters(
        nv.t.submit(
            TPM_CC_NV_DEFINE_SPACE,
            command_packet(TPM_CC_NV_DEFINE_SPACE, (TPM_RH_OWNER,), params, auth=True),
        )
    )


def read(nv: NvProbe, handle: int) -> int:
    packet = command_packet(0x14E, (handle, handle), u16(8) + u16(0), auth=True)
    data = response_parameters(nv.t.submit(0x14E, packet))
    value, _ = read_tpm2b(data)
    return int.from_bytes(value, "big")


def policy_nv(nv: NvProbe, session: int, handle: int, value: int) -> None:
    params = tpm2b(value.to_bytes(8, "big")) + u16(0) + u16(0)
    packet = command_packet(0x149, (handle, handle, session), params, auth=True)
    response_parameters(nv.t.submit(0x149, packet))


def policy_cp_hash(nv: NvProbe, session: int, value: bytes) -> None:
    packet = command_packet(TPM_CC_POLICY_CP_HASH, (session,), tpm2b(value), auth=False)
    response_parameters(nv.t.submit(TPM_CC_POLICY_CP_HASH, packet))


def increment_policy(nv: NvProbe, handle: int, session: int) -> None:
    packet = policy_auth_packet(TPM_CC_NV_INCREMENT, (handle, handle), session)
    response_parameters(nv.t.submit(TPM_CC_NV_INCREMENT, packet))


RESOURCE_NAMES = (
    "bootstrap_recovery_session",
    "normal_session",
    "negative_cross_branch_session",
    "negative_reversed_or_session",
    "psa_external_key",
    "recovery_external_key",
)


def mandatory_cleanup(
    evidence: dict[str, Any],
    nv: NvProbe | None,
    transport: TbsTransport,
    resources: dict[str, int | None],
    selected_nv: int | None,
    nv_created: bool,
    tbs_created: bool,
) -> bool:
    """Clean every semantic resource and return false on any cleanup failure."""
    cleanup = evidence["cleanup"]
    cleanup_handles = evidence["cleanup_handles"]
    all_passed = True
    for name in RESOURCE_NAMES:
        handle = resources[name]
        if handle is None:
            cleanup[name] = "NOT_CREATED"
            cleanup_handles[name] = None
            continue
        cleanup_handles[name] = f"0x{handle:08X}"
        try:
            if nv is None:
                raise ProbeError("CLEANUP_CONTEXT_MISSING", name)
            nv.flush(handle)
            cleanup[name] = "PASS"
        except (ProbeError, OSError):
            cleanup[name] = "FAIL"
            all_passed = False

    if not nv_created or selected_nv is None:
        cleanup["nv"] = "NOT_CREATED"
        cleanup_handles["nv"] = None
    else:
        cleanup_handles["nv"] = f"0x{selected_nv:08X}"
        try:
            if nv is None:
                raise ProbeError("CLEANUP_CONTEXT_MISSING", "nv")
            nv.undefine(selected_nv)
            cleanup["nv"] = "PASS"
        except (ProbeError, OSError):
            cleanup["nv"] = "FAIL"
            all_passed = False

    if not tbs_created:
        cleanup["tbs_context"] = "NOT_CREATED"
    else:
        try:
            transport.close()
            cleanup["tbs_context"] = "PASS"
        except (ProbeError, OSError):
            cleanup["tbs_context"] = "FAIL"
            all_passed = False
    return all_passed


def apply_cleanup_result(evidence: dict[str, Any], result: int, cleanup_passed: bool) -> int:
    """Make mandatory cleanup authoritative for the final process result."""
    if cleanup_passed:
        return result
    evidence["failure"] = {
        "reason": "CLEANUP_FAILED",
        "detail": "one or more mandatory cleanup operations failed",
    }
    return 1


def run(output: Path) -> int:
    evidence: dict[str, Any] = {
        "schema": "CryptoHunter.Stage9TwoBranchPhysicalProbeV1",
        "test_only": True,
        "normal": "NOT_RUN",
        "bootstrap_recovery": "NOT_RUN",
        "negative_cross_branch": "NOT_RUN",
        "negative_reversed_or": "NOT_RUN",
        "commands": [],
        "responses": [],
        "cleanup": {name: "NOT_CREATED" for name in (*RESOURCE_NAMES, "nv", "tbs_context")},
        "cleanup_handles": {name: None for name in (*RESOURCE_NAMES, "nv")},
    }
    t = TbsTransport(evidence)
    selected: int | None = None
    created = False
    tbs_created = False
    nv: NvProbe | None = None
    result = 1
    resources: dict[str, int | None] = {name: None for name in RESOURCE_NAMES}
    try:
        if os.name != "nt" or platform.system() != "Windows":
            raise ProbeError("PHYSICAL_WINDOWS_11_REQUIRED", "physical Windows only")
        qualify_windows(evidence)
        t.open()
        tbs_created = True
        nv = NvProbe(t)
        selected = _choose_unused(nv.handles())
        private_psa, private_recovery = (
            ec.generate_private_key(ec.SECP256R1()),
            ec.generate_private_key(ec.SECP256R1()),
        )
        psa_point = private_psa.public_key().public_numbers()
        psa_public = ecc_public_area(
            psa_point.x.to_bytes(32, "big"), psa_point.y.to_bytes(32, "big")
        )
        psa_handle, psa_name = nv.load_external(psa_public, TPM_RH_OWNER)
        resources["psa_external_key"] = psa_handle
        recovery_point = private_recovery.public_key().public_numbers()
        recovery_public = ecc_public_area(
            recovery_point.x.to_bytes(32, "big"), recovery_point.y.to_bytes(32, "big")
        )
        recovery_handle, recovery_name = nv.load_external(recovery_public, TPM_RH_OWNER)
        resources["recovery_external_key"] = recovery_handle
        normal_branch, recovery_branch = (
            branch(psa_name, REF_NORMAL),
            branch(recovery_name, REF_RECOVERY),
        )
        root = h(bytes(32), u32(TPM_CC_POLICY_OR), normal_branch, recovery_branch)
        define(nv, selected, root)
        created = True
        pre_public, pre_name = nv.read_public(selected)
        s, *_ = nv.start_policy()
        resources["bootstrap_recovery_session"] = s
        nv.policy_command_code(s, TPM_CC_NV_INCREMENT)
        policy_cp_hash(nv, s, cp_hash(pre_name))
        approved = nv.policy_digest(s)
        sign_authorize(
            nv, s, approved, REF_RECOVERY, private_recovery, recovery_handle, recovery_name
        )
        nv.policy_or(s, [normal_branch, recovery_branch])
        increment_policy(nv, selected, s)
        evidence["bootstrap_recovery"] = "PASS"
        initialized = read(nv, selected)
        _, post_name = nv.read_public(selected)
        s2, *_ = nv.start_policy()
        resources["normal_session"] = s2
        policy_nv(nv, s2, selected, initialized)
        nv.policy_command_code(s2, TPM_CC_NV_INCREMENT)
        approved_normal = nv.policy_digest(s2)
        sign_authorize(nv, s2, approved_normal, REF_NORMAL, private_psa, psa_handle, psa_name)
        nv.policy_or(s2, [normal_branch, recovery_branch])
        increment_policy(nv, selected, s2)
        evidence["normal"] = "PASS"
        # Wrong authority ticket cannot satisfy PolicyAuthorize for K_PSA.Name.
        s3, *_ = nv.start_policy()
        resources["negative_cross_branch_session"] = s3
        policy_nv(nv, s3, selected, initialized + 1)
        nv.policy_command_code(s3, TPM_CC_NV_INCREMENT)
        wrong_approved = nv.policy_digest(s3)
        try:
            sign_authorize(
                nv, s3, wrong_approved, REF_NORMAL, private_recovery, recovery_handle, psa_name
            )
        except ProbeError:
            evidence["negative_cross_branch"] = "PASS"
        else:
            raise ProbeError(
                "CROSS_BRANCH_NOT_REJECTED", "recovery ticket accepted as normal authority"
            )
        s4, *_ = nv.start_policy()
        resources["negative_reversed_or_session"] = s4
        policy_nv(nv, s4, selected, initialized + 1)
        nv.policy_command_code(s4, TPM_CC_NV_INCREMENT)
        reversed_approved = nv.policy_digest(s4)
        sign_authorize(nv, s4, reversed_approved, REF_NORMAL, private_psa, psa_handle, psa_name)
        nv.policy_or(s4, [recovery_branch, normal_branch])
        try:
            increment_policy(nv, selected, s4)
        except ProbeError:
            evidence["negative_reversed_or"] = "PASS"
        else:
            raise ProbeError(
                "REVERSED_POLICY_OR_NOT_REJECTED", "wrong ordered PolicyOR authorized increment"
            )
        evidence.update(
            final_generation=read(nv, selected),
            pre_write_name=pre_name.hex(),
            post_write_name=post_name.hex(),
            root_policy=root.hex(),
        )
        result = 0
    except ProbeError as exc:
        evidence["failure"] = {"reason": exc.reason, "detail": exc.detail}
    except (OSError, ValueError) as exc:
        evidence["failure"] = {
            "reason": "UNEXPECTED_PROBE_FAILURE",
            "detail": f"{type(exc).__name__}: {exc}",
        }
    finally:
        cleanup_passed = mandatory_cleanup(
            evidence, nv, t, resources, selected, created, tbs_created
        )
    result = apply_cleanup_result(evidence, result, cleanup_passed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return run(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
