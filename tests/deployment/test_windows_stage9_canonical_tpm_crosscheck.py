from __future__ import annotations

import hashlib
import struct
from copy import deepcopy

import pytest

from deployment.windows_stage9_canonical_tpm_crosscheck import (
    assert_checkpoint,
    apply_cleanup_failure,
    load_recovery_test_key,
    load_psa_test_key,
    load_inputs,
    require_incremented_generation,
    runtime_vector_for_observed_generation,
    start_trial,
    validate_test_only_inputs,
    validate_test_private_keys,
)
from deployment.windows_tpm_substrate_probe import (
    ProbeError,
    TPM_CC_START_AUTH_SESSION,
    TPM_RH_NULL,
    TPM_RH_OWNER,
)


class FakeLoadExternal:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, int]] = []

    def load_external(self, public: bytes, hierarchy: int) -> tuple[int, bytes]:
        self.calls.append((public, hierarchy))
        return 0x80000000, b"name"


class FakeTrialTransport:
    def __init__(self) -> None:
        self.request = b""

    def submit(self, code: int, request: bytes) -> bytes:
        assert code == TPM_CC_START_AUTH_SESSION
        self.request = request
        body = struct.pack(">I", 0x03000000) + b"\x00\x01n"
        return struct.pack(">HII", 0x8001, 10 + len(body), 0) + body


class FakeTrialNv:
    def __init__(self) -> None:
        self.t = FakeTrialTransport()


def test_deterministic_test_scalars_match_canonical_publics() -> None:
    release, enrollment, _, keys = load_inputs()
    psa, recovery = validate_test_private_keys(release, enrollment, keys)
    assert psa.private_numbers().private_value == 1
    assert recovery.private_numbers().private_value == 2


def test_real_bootstrap_recovery_key_is_loaded_under_owner() -> None:
    nv = FakeLoadExternal()
    assert load_recovery_test_key(nv, b"public") == (0x80000000, b"name")  # type: ignore[arg-type]
    assert nv.calls == [(b"public", TPM_RH_OWNER)]


def test_real_normal_psa_key_is_loaded_under_owner() -> None:
    nv = FakeLoadExternal()
    assert load_psa_test_key(nv, b"public") == (0x80000000, b"name")  # type: ignore[arg-type]
    assert nv.calls == [(b"public", TPM_RH_OWNER)]


def test_trial_session_still_uses_null_tpm_key_and_bind_handles() -> None:
    nv = FakeTrialNv()
    assert start_trial(nv) == 0x03000000  # type: ignore[arg-type]
    assert nv.t.request[10:18] == struct.pack(">II", TPM_RH_NULL, TPM_RH_NULL)


def test_crosscheck_uses_release_policy_refs_not_old_probe_refs() -> None:
    release, _, vector, _ = load_inputs()
    refs = release["policy_refs"]
    assert refs["normal_label"] == "CryptoHunter.Stage9.NV.Normal.v1"
    assert refs["recovery_label"] == "CryptoHunter.Stage9.NV.BootstrapRecovery.v1"
    assert (
        vector["normal_policy_ref"]
        == hashlib.sha256(refs["normal_label"].encode("ascii")).hexdigest()
    )
    assert (
        vector["recovery_policy_ref"]
        == hashlib.sha256(refs["recovery_label"].encode("ascii")).hexdigest()
    )


def test_observed_generation_11_does_not_mutate_fixture_generation_7() -> None:
    release, enrollment, fixture, _ = load_inputs()
    runtime_enrollment, runtime = runtime_vector_for_observed_generation(release, enrollment, 11)
    assert fixture["normal_generation"] == "7"
    assert enrollment["vector_generation"] == "7"
    assert runtime_enrollment["vector_generation"] == "11"
    assert runtime["normal_generation"] == "11"
    assert runtime["normal_approved_policy"] != fixture["normal_approved_policy"]
    assert runtime["normal_branch_digest"] == fixture["normal_branch_digest"]
    assert runtime["production_root_policy_digest"] == fixture["production_root_policy_digest"]
    observed_final_generation = 11 + 1
    require_incremented_generation(11, observed_final_generation)
    assert observed_final_generation == 12


@pytest.mark.parametrize(
    "mutation",
    (
        "release_production",
        "enrollment_production",
        "recovery_production",
        "psa_production",
        "keys_production",
    ),
)
def test_disposable_runner_rejects_non_test_only_authority_material(mutation: str) -> None:
    release, enrollment, _, keys = load_inputs()
    release, enrollment, keys = deepcopy(release), deepcopy(enrollment), deepcopy(keys)
    if mutation == "release_production":
        release["purpose"] = "PRODUCTION"
    elif mutation == "enrollment_production":
        enrollment["purpose"] = "PRODUCTION"
    elif mutation == "recovery_production":
        release["k_recovery"]["provenance"] = "PRODUCTION_PROVISIONED"
    elif mutation == "psa_production":
        enrollment["k_psa"]["provenance"] = "PRODUCTION_PROVISIONED"
    else:
        keys["purpose"] = "PRODUCTION"
    with pytest.raises(ProbeError, match="PRODUCTION_MATERIAL_FORBIDDEN"):
        validate_test_only_inputs(release, enrollment, keys)


def test_checkpoint_is_fail_closed_and_records_all_three_sources() -> None:
    evidence = {"checks": {}}
    assert_checkpoint(evidence, "same", b"\xaa" * 32, "aa" * 32)
    assert evidence["checks"]["same"] == {
        "tpm": "aa" * 32,
        "canonical_generator": "aa" * 32,
        "independent_verifier": "aa" * 32,
        "status": "PASS",
    }
    with pytest.raises(ProbeError, match="CANONICAL_TPM_DIGEST_MISMATCH"):
        assert_checkpoint(evidence, "different", b"\xbb" * 32, "aa" * 32)


def test_cleanup_failure_is_primary_only_without_earlier_failure() -> None:
    evidence = {"cleanup": {"nv": "FAIL", "tbs_context": "PASS"}}
    apply_cleanup_failure(evidence)
    assert evidence["failure"] == {
        "reason": "CLEANUP_FAILED",
        "detail": "mandatory cleanup failed: nv",
    }


def test_cleanup_failure_preserves_earlier_primary_failure() -> None:
    evidence = {
        "cleanup": {"nv": "FAIL"},
        "failure": {"reason": "CANONICAL_TPM_DIGEST_MISMATCH", "detail": "root"},
    }
    apply_cleanup_failure(evidence)
    assert evidence["failure"]["reason"] == "CANONICAL_TPM_DIGEST_MISMATCH"
    assert evidence["failure"]["primary_reason"] == "CANONICAL_TPM_DIGEST_MISMATCH"
    assert evidence["failure"]["cleanup_failure"] is True
    assert evidence["failure"]["cleanup_failure_resources"] == ["nv"]
