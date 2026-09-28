"""Disposable physical TPM cross-check of the canonical Stage-9 TEST vector.

This runner consumes the checked-in TEST_ONLY vector and deterministic test
keys.  It neither creates nor represents production authority material.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
from typing import Any

from cryptography.hazmat.primitives.asymmetric import ec

from deployment.windows_stage9_policy_vector import generate_policy_vector
from deployment.windows_stage9_policy_vector_verifier import verify_policy_vector
from deployment.windows_stage9_two_branch_probe import (
    define,
    increment_policy,
    policy_cp_hash,
    policy_nv,
    read,
    sign_authorize,
)
from deployment.windows_tpm_substrate_probe import (
    NvProbe,
    ProbeError,
    TbsTransport,
    TPM_ALG_NULL,
    TPM_ALG_SHA256,
    TPM_CC_START_AUTH_SESSION,
    TPM_RH_NULL,
    TPM_RH_OWNER,
    command_packet,
    qualify_windows,
    read_tpm2b,
    read_u32,
    response_parameters,
    tpm2b,
    u16,
)

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/windows_stage9_policy_vector_v1.json"
TEST_KEYS = (
    Path(__file__).resolve().parents[1]
    / "tests/fixtures/windows_stage9_policy_vector_v1_test_keys.json"
)
TPM_SE_TRIAL = 0x03


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    keys = json.loads(TEST_KEYS.read_text(encoding="utf-8"))
    values = (
        document["release_policy"],
        document["enrollment_policy_material"],
        document["policy_vector"],
        keys,
    )
    validate_test_only_inputs(values[0], values[1], values[3])
    return values


def validate_test_only_inputs(
    release: dict[str, Any], enrollment: dict[str, Any], keys: dict[str, Any]
) -> None:
    """Enforce the disposable runner's authority boundary before opening TBS."""
    if (
        release.get("purpose") != "TEST_ONLY"
        or enrollment.get("purpose") != "TEST_ONLY"
        or release.get("k_recovery", {}).get("provenance") != "TEST_FIXTURE"
        or enrollment.get("k_psa", {}).get("provenance") != "TEST_FIXTURE"
        or keys.get("purpose") != "TEST_ONLY"
    ):
        raise ProbeError("PRODUCTION_MATERIAL_FORBIDDEN", "cross-check accepts TEST_ONLY fixtures")


def validate_test_private_keys(
    release: dict[str, Any], enrollment: dict[str, Any], keys: dict[str, Any]
) -> tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePrivateKey]:
    """Prove that the explicitly TEST-only scalars match both canonical publics."""
    private = tuple(
        ec.derive_private_key(int(keys[field], 16), ec.SECP256R1())
        for field in ("k_psa_private_scalar_hex", "k_recovery_private_scalar_hex")
    )
    for key, public_hex in zip(
        private,
        (enrollment["k_psa"]["public_hex"], release["k_recovery"]["public_hex"]),
        strict=True,
    ):
        numbers = key.public_key().public_numbers()
        raw = bytes.fromhex(public_hex)
        coordinates = tpm2b(numbers.x.to_bytes(32, "big")) + tpm2b(
            numbers.y.to_bytes(32, "big")
        )
        if not raw.endswith(coordinates):
            raise ProbeError("TEST_PRIVATE_KEY_MISMATCH", "scalar does not match canonical TPMT_PUBLIC")
    return private  # type: ignore[return-value]


def assert_checkpoint(evidence: dict[str, Any], field: str, produced: bytes, expected: str) -> None:
    evidence["checks"][field] = {
        "tpm": produced.hex(),
        "canonical_generator": expected,
        "independent_verifier": expected,
        "status": "PASS" if produced.hex() == expected else "FAIL",
    }
    if produced.hex() != expected:
        raise ProbeError("CANONICAL_TPM_DIGEST_MISMATCH", field)


def start_trial(nv: NvProbe) -> int:
    params = tpm2b(secrets.token_bytes(32)) + tpm2b(b"") + bytes((TPM_SE_TRIAL,)) + u16(
        TPM_ALG_NULL
    ) + u16(TPM_ALG_SHA256)
    data = response_parameters(
        nv.t.submit(
            TPM_CC_START_AUTH_SESSION,
            command_packet(TPM_CC_START_AUTH_SESSION, (TPM_RH_NULL, TPM_RH_NULL), params, auth=False),
        )
    )
    session, offset = read_u32(data)
    read_tpm2b(data, offset)
    return session


def load_recovery_test_key(nv: NvProbe, public: bytes) -> tuple[int, bytes]:
    """Load the real bootstrap verifier Owner-backed for a non-NULL ticket."""
    return nv.load_external(public, TPM_RH_OWNER)


def load_psa_test_key(nv: NvProbe, public: bytes) -> tuple[int, bytes]:
    """Load the real normal-branch verifier Owner-backed for a non-NULL ticket."""
    return nv.load_external(public, TPM_RH_OWNER)


def runtime_vector_for_observed_generation(
    release: dict[str, Any], enrollment: dict[str, Any], generation: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a non-frozen TEST vector for the counter value actually observed."""
    if generation < 0 or generation >= (1 << 64):
        raise ProbeError("OBSERVED_GENERATION_OVERFLOW", str(generation))
    runtime_enrollment = deepcopy(enrollment)
    runtime_enrollment["vector_generation"] = str(generation)
    runtime_vector = generate_policy_vector(release, runtime_enrollment)
    verify_policy_vector(release, runtime_enrollment, runtime_vector)
    return runtime_enrollment, runtime_vector


def require_incremented_generation(initial: int, final: int) -> None:
    if initial == (1 << 64) - 1:
        raise ProbeError("OBSERVED_GENERATION_OVERFLOW", str(initial))
    if final != initial + 1:
        raise ProbeError(
            "FINAL_GENERATION_MISMATCH", f"expected={initial + 1}, actual={final}"
        )


def trial_authorize(nv: NvProbe, session: int, approved: bytes, ref: bytes, name: bytes) -> None:
    # In a trial session the TPM deliberately does not validate the ticket.
    from deployment.windows_tpm_substrate_probe import VerificationTicket, TPM_ST_VERIFIED

    nv.policy_authorize(session, approved, ref, name, VerificationTicket(TPM_ST_VERIFIED, TPM_RH_NULL, b""))


def apply_cleanup_failure(evidence: dict[str, Any]) -> None:
    failed = sorted(name for name, status in evidence["cleanup"].items() if status == "FAIL")
    if not failed:
        return
    detail = "mandatory cleanup failed: " + ", ".join(failed)
    if "failure" in evidence:
        evidence["failure"]["primary_reason"] = evidence["failure"]["reason"]
        evidence["failure"]["cleanup_failure"] = True
        evidence["failure"]["cleanup_failure_resources"] = failed
    else:
        evidence["failure"] = {"reason": "CLEANUP_FAILED", "detail": detail}


def run(output: Path) -> int:
    evidence: dict[str, Any] = {
        "schema": "CryptoHunter.Stage9CanonicalVectorPhysicalCrossCheckV1",
        "test_only": True,
        "checks": {},
        "cleanup": {},
        "commands": [],
        "responses": [],
    }
    transport = TbsTransport(evidence)
    nv: NvProbe | None = None
    sessions: list[int] = []
    recovery_handle: int | None = None
    psa_handle: int | None = None
    index: int | None = None
    nv_created = False
    tbs_created = False
    result = 1
    try:
        if os.name != "nt" or platform.system() != "Windows":
            raise ProbeError("PHYSICAL_WINDOWS_11_REQUIRED", "physical Windows only")
        release, enrollment, canonical, keys = load_inputs()
        generated = generate_policy_vector(release, enrollment)
        if generated != canonical:
            raise ProbeError("GENERATOR_FIXTURE_MISMATCH", "canonical fixture is stale")
        verify_policy_vector(release, enrollment, canonical)
        psa_private, recovery_private = validate_test_private_keys(release, enrollment, keys)
        qualify_windows(evidence)
        transport.open()
        tbs_created = True
        nv = NvProbe(transport)
        index = int(release["nv_template"]["index"], 16)
        if index in nv.handles():
            raise ProbeError("CANONICAL_NV_INDEX_IN_USE", f"0x{index:08X}")
        recovery_public = bytes.fromhex(release["k_recovery"]["public_hex"])
        recovery_handle, recovery_name = load_recovery_test_key(nv, recovery_public)
        psa_public = bytes.fromhex(enrollment["k_psa"]["public_hex"])
        psa_handle, psa_name = load_psa_test_key(nv, psa_public)
        evidence["verification_key_hierarchy"] = "TPM_RH_OWNER"
        evidence["verification_key_hierarchy_raw"] = f"0x{TPM_RH_OWNER:08X}"
        define(nv, index, bytes.fromhex(canonical["production_root_policy_digest"]))
        nv_created = True
        pre_public, pre_name = nv.read_public(index)
        assert_checkpoint(evidence, "nv_public_pre_write", pre_public, canonical["nv_public_pre_write"])
        assert_checkpoint(evidence, "nv_name_pre_write", pre_name, canonical["nv_name_pre_write"])

        normal_ref = bytes.fromhex(canonical["normal_policy_ref"])
        recovery_ref = bytes.fromhex(canonical["recovery_policy_ref"])
        branches = [bytes.fromhex(canonical[x]) for x in ("normal_branch_digest", "recovery_branch_digest")]

        bootstrap = nv.start_policy()[0]
        sessions.append(bootstrap)
        nv.policy_command_code(bootstrap, 0x134)
        policy_cp_hash(nv, bootstrap, bytes.fromhex(canonical["bootstrap_increment_cp_hash"]))
        approved = nv.policy_digest(bootstrap)
        assert_checkpoint(evidence, "bootstrap_approved_policy", approved, canonical["bootstrap_approved_policy"])
        ticket = sign_authorize(
            nv,
            bootstrap,
            approved,
            recovery_ref,
            recovery_private,
            recovery_handle,
            recovery_name,
        )
        evidence["verification_ticket"] = {
            "tag": "TPM_ST_VERIFIED",
            "tag_raw": f"0x{ticket.tag:04X}",
            "hierarchy": "TPM_RH_OWNER",
            "hierarchy_raw": f"0x{ticket.hierarchy:08X}",
            "digest_size": len(ticket.digest),
        }
        assert_checkpoint(evidence, "recovery_branch_digest", nv.policy_digest(bootstrap), canonical["recovery_branch_digest"])
        nv.policy_or(bootstrap, branches)
        assert_checkpoint(evidence, "final_ordered_policy_or_root", nv.policy_digest(bootstrap), canonical["production_root_policy_digest"])
        increment_policy(nv, index, bootstrap)
        post_public, post_name = nv.read_public(index)
        assert_checkpoint(evidence, "nv_public_post_write", post_public, canonical["nv_public_post_write"])
        assert_checkpoint(evidence, "nv_name_post_write", post_name, canonical["nv_name_post_write"])
        if pre_name == post_name:
            raise ProbeError("NV_NAME_DID_NOT_CHANGE", pre_name.hex())
        observed_generation = read(nv, index)
        if observed_generation == (1 << 64) - 1:
            raise ProbeError("OBSERVED_GENERATION_OVERFLOW", str(observed_generation))
        _, runtime_vector = runtime_vector_for_observed_generation(
            release, enrollment, observed_generation
        )
        evidence.update(
            fixture_generation=int(canonical["normal_generation"]),
            fixture_generation_mode="TRIAL_DIGEST_ONLY",
            observed_initial_generation=observed_generation,
            runtime_normal_generation=observed_generation,
            fixture_normal_approved_policy=canonical["normal_approved_policy"],
            runtime_normal_approved_policy=runtime_vector["normal_approved_policy"],
            pre_write_name=pre_name.hex(),
            post_write_name=post_name.hex(),
            runtime_normal_branch_digest=runtime_vector["normal_branch_digest"],
            runtime_root_digest=runtime_vector["production_root_policy_digest"],
            runtime_vector_classification=[
                "RUNTIME_TEST_ONLY",
                "OBSERVED_PHYSICAL_GENERATION",
                "NOT_FOR_FREEZE",
            ],
        )

        post_trial = start_trial(nv)
        sessions.append(post_trial)
        nv.policy_command_code(post_trial, 0x134)
        policy_cp_hash(nv, post_trial, bytes.fromhex(canonical["post_write_recovery_increment_cp_hash"]))
        assert_checkpoint(evidence, "post_write_recovery_approved_policy", nv.policy_digest(post_trial), canonical["post_write_recovery_approved_policy"])

        normal = start_trial(nv)
        sessions.append(normal)
        policy_nv(nv, normal, index, int(canonical["normal_generation"]))
        nv.policy_command_code(normal, 0x134)
        assert_checkpoint(evidence, "normal_approved_policy", nv.policy_digest(normal), canonical["normal_approved_policy"])
        trial_authorize(nv, normal, nv.policy_digest(normal), normal_ref, bytes.fromhex(canonical["k_psa_name"]))
        assert_checkpoint(evidence, "normal_branch_digest", nv.policy_digest(normal), canonical["normal_branch_digest"])
        nv.policy_or(normal, branches)
        assert_checkpoint(evidence, "normal_final_ordered_policy_or_root", nv.policy_digest(normal), canonical["production_root_policy_digest"])

        real_normal = nv.start_policy()[0]
        sessions.append(real_normal)
        policy_nv(nv, real_normal, index, observed_generation)
        nv.policy_command_code(real_normal, 0x134)
        real_approved = nv.policy_digest(real_normal)
        assert_checkpoint(
            evidence,
            "runtime_normal_approved_policy",
            real_approved,
            runtime_vector["normal_approved_policy"],
        )
        sign_authorize(
            nv,
            real_normal,
            real_approved,
            normal_ref,
            psa_private,
            psa_handle,
            psa_name,
        )
        assert_checkpoint(
            evidence,
            "runtime_normal_branch_digest",
            nv.policy_digest(real_normal),
            runtime_vector["normal_branch_digest"],
        )
        nv.policy_or(real_normal, branches)
        assert_checkpoint(
            evidence,
            "runtime_final_ordered_policy_or_root",
            nv.policy_digest(real_normal),
            runtime_vector["production_root_policy_digest"],
        )
        increment_policy(nv, index, real_normal)
        final_generation = read(nv, index)
        evidence["observed_final_generation"] = final_generation
        require_incremented_generation(observed_generation, final_generation)
        evidence["name_lifecycle"] = "PASS"
        result = 0
    except (ProbeError, OSError, ValueError, KeyError) as exc:
        evidence["failure"] = {"reason": getattr(exc, "reason", type(exc).__name__), "detail": getattr(exc, "detail", str(exc))}
    finally:
        if nv is not None:
            for session in sessions:
                try:
                    nv.flush(session)
                    evidence["cleanup"][f"session_0x{session:08X}"] = "PASS"
                except (ProbeError, OSError):
                    evidence["cleanup"][f"session_0x{session:08X}"] = "FAIL"
                    result = 1
            if recovery_handle is not None:
                try:
                    nv.flush(recovery_handle)
                    evidence["cleanup"]["recovery_external_key"] = "PASS"
                except (ProbeError, OSError):
                    evidence["cleanup"]["recovery_external_key"] = "FAIL"
                    result = 1
            if psa_handle is not None:
                try:
                    nv.flush(psa_handle)
                    evidence["cleanup"]["psa_external_key"] = "PASS"
                except (ProbeError, OSError):
                    evidence["cleanup"]["psa_external_key"] = "FAIL"
                    result = 1
            if index is not None and nv_created:
                try:
                    nv.undefine(index)
                    evidence["cleanup"]["nv"] = "PASS"
                except (ProbeError, OSError):
                    evidence["cleanup"]["nv"] = "FAIL"
                    result = 1
        if tbs_created:
            try:
                transport.close()
                evidence["cleanup"]["tbs_context"] = "PASS"
            except (ProbeError, OSError):
                evidence["cleanup"]["tbs_context"] = "FAIL"
                result = 1
        apply_cleanup_failure(evidence)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return run(parser.parse_args(argv).output)


if __name__ == "__main__":
    raise SystemExit(main())
