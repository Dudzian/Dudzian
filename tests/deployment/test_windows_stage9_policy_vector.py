from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import pytest
from deployment.windows_stage9_policy_material import (
    PolicyVectorError,
    canonical_digest,
    canonical_json_bytes,
)
from deployment.windows_stage9_policy_vector import generate_policy_vector
from deployment.windows_stage9_policy_vector_verifier import verify_policy_vector

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/windows_stage9_policy_vector_v1.json"


def material():
    doc = json.loads(FIXTURE.read_text())
    return doc["release_policy"], doc["enrollment_policy_material"], doc["policy_vector"]


def rejects(release, enrollment, vector=None):
    with pytest.raises(PolicyVectorError):
        (
            verify_policy_vector(release, enrollment, vector)
            if vector is not None
            else generate_policy_vector(release, enrollment)
        )


def mutate_public(key, offset, replacement):
    raw = bytearray.fromhex(key["public_hex"])
    raw[offset : offset + len(replacement)] = replacement
    key["public_hex"] = raw.hex()


def test_schema_generator_fixture_and_independent_verifier():
    release, enrollment, expected = material()
    actual = generate_policy_vector(release, enrollment)
    assert actual == expected
    verify_policy_vector(release, enrollment, actual)


def test_authority_boundary_places_psa_only_in_enrollment():
    release, enrollment, _ = material()
    assert "k_psa" not in release
    assert enrollment["k_psa"]["binding_authority"] == "PDSA_ENROLLMENT_PACKAGE_V1"
    assert enrollment["release_policy_digest"] == canonical_digest(release).hex()


@pytest.mark.parametrize(
    ("purpose", "provenance"),
    [("PRODUCTION", "TEST_FIXTURE"), ("TEST_ONLY", "PRODUCTION_PROVISIONED")],
)
def test_release_purpose_provenance_is_fail_closed(purpose, provenance):
    release, enrollment, _ = material()
    release["purpose"] = purpose
    enrollment["purpose"] = purpose
    release["k_recovery"]["provenance"] = provenance
    enrollment["k_psa"]["provenance"] = provenance
    enrollment["release_policy_digest"] = canonical_digest(release).hex()
    rejects(release, enrollment)


@pytest.mark.parametrize(
    ("purpose", "provenance"),
    [("PRODUCTION", "TEST_FIXTURE"), ("TEST_ONLY", "PRODUCTION_PROVISIONED")],
)
def test_enrollment_purpose_provenance_is_fail_closed(purpose, provenance):
    release, enrollment, _ = material()
    release["purpose"] = purpose
    enrollment["purpose"] = purpose
    expected = "PRODUCTION_PROVISIONED" if purpose == "PRODUCTION" else "TEST_FIXTURE"
    release["k_recovery"]["provenance"] = expected
    enrollment["k_psa"]["provenance"] = provenance
    enrollment["release_policy_digest"] = canonical_digest(release).hex()
    rejects(release, enrollment)


def test_policy_refs_are_sha256_ascii_labels():
    release, enrollment, _ = material()
    release["policy_refs"]["normal_hex"] = "00" * 32
    enrollment["release_policy_digest"] = canonical_digest(release).hex()
    rejects(release, enrollment)


def test_nv_name_lifecycle_changes_dynamic_values_not_root():
    release, enrollment, vector = material()
    assert vector["nv_name_pre_write"] != vector["nv_name_post_write"]
    assert vector["bootstrap_increment_cp_hash"] != vector["post_write_recovery_increment_cp_hash"]
    assert vector["bootstrap_approved_policy"] != vector["post_write_recovery_approved_policy"]
    assert (
        generate_policy_vector(release, enrollment)["production_root_policy_digest"]
        == vector["production_root_policy_digest"]
    )


@pytest.mark.parametrize(("offset", "replacement"), [(2, b"\x00\x04"), (0, b"\x00\x01")])
def test_tpmt_public_rejects_wrong_name_alg_or_type(offset, replacement):
    release, enrollment, _ = material()
    mutate_public(enrollment["k_psa"], offset, replacement)
    rejects(release, enrollment)


def test_tpmt_public_rejects_wrong_curve():
    release, enrollment, _ = material()
    mutate_public(enrollment["k_psa"], 48, b"\x00\x04")
    rejects(release, enrollment)


def test_tpmt_public_rejects_wrong_scheme():
    release, enrollment, _ = material()
    mutate_public(enrollment["k_psa"], 44, b"\x00\x14")
    rejects(release, enrollment)


def test_tpmt_public_rejects_wrong_scheme_hash():
    release, enrollment, _ = material()
    mutate_public(enrollment["k_psa"], 46, b"\x00\x04")
    rejects(release, enrollment)


@pytest.mark.parametrize("mutation", ["truncate", "trailing", "coordinate"])
def test_tpmt_public_rejects_bad_extent_or_coordinate(mutation):
    release, enrollment, _ = material()
    key = enrollment["k_psa"]
    raw = bytearray.fromhex(key["public_hex"])
    if mutation == "truncate":
        raw = raw[:-1]
    elif mutation == "trailing":
        raw += b"\x00"
    else:
        raw[52:54] = b"\x00\x21"
    key["public_hex"] = raw.hex()
    rejects(release, enrollment)


def test_wrong_metadata_is_rejected_even_with_valid_bytes():
    release, enrollment, _ = material()
    enrollment["k_psa"]["name_algorithm"] = "TPM_ALG_SHA1"
    rejects(release, enrollment)


@pytest.mark.parametrize(
    "change",
    ["reversed", "duplicate", "key_name", "policy_ref", "template", "cp_hash", "name_swap"],
)
def test_vector_or_source_tampering_is_rejected(change):
    release, enrollment, vector = material()
    if change == "reversed":
        vector["branch_order"].reverse()
    elif change == "duplicate":
        vector["recovery_branch_digest"] = vector["normal_branch_digest"]
    elif change == "key_name":
        vector["k_psa_name"] = "00" * 34
    elif change == "policy_ref":
        vector["normal_policy_ref"] = "00" * 32
    elif change == "template":
        release["nv_template"]["attributes"] = "0204001a"
        enrollment["release_policy_digest"] = canonical_digest(release).hex()
    elif change == "cp_hash":
        vector["bootstrap_increment_cp_hash"] = vector["post_write_recovery_increment_cp_hash"]
    else:
        vector["nv_name_pre_write"], vector["nv_name_post_write"] = (
            vector["nv_name_post_write"],
            vector["nv_name_pre_write"],
        )
    rejects(release, enrollment, vector)


def test_semantically_identical_branches_are_rejected():
    release, enrollment, _ = material()
    enrollment["k_psa"]["public_hex"] = release["k_recovery"]["public_hex"]
    rejects(release, enrollment)


def test_generation_is_bound_to_enrollment_even_if_vector_is_self_consistent():
    release, enrollment, vector = material()
    vector["normal_generation"] = "8"
    post = bytes.fromhex(vector["nv_name_post_write"])
    h = lambda x: hashlib.sha256(x).digest()
    z = bytes(32)
    args = h(struct.pack(">Q", 8) + b"\x00\x00\x00\x00")
    d = h(z + struct.pack(">I", 0x149) + args + post)
    vector["normal_approved_policy"] = h(
        d + struct.pack(">I", 0x16C) + struct.pack(">I", 0x134)
    ).hex()
    rejects(release, enrollment, vector)


def test_canonical_serialization_ignores_input_order_and_whitespace():
    release, _, _ = material()
    reordered = json.loads(
        json.dumps(release, indent=7), object_pairs_hook=lambda pairs: dict(reversed(pairs))
    )
    assert canonical_json_bytes(reordered) == canonical_json_bytes(release)
    assert canonical_digest(reordered) == canonical_digest(release)


def test_canonical_serialization_utf8_and_number_edges():
    assert (
        canonical_json_bytes({"z": "zażółć", "a": 9007199254740991})
        == b'{"a":9007199254740991,"z":"za\xc5\xbc\xc3\xb3\xc5\x82\xc4\x87"}'
    )
    with pytest.raises(PolicyVectorError):
        canonical_json_bytes({"n": 9007199254740992})
    with pytest.raises(PolicyVectorError):
        canonical_json_bytes({"n": 1.0})


def test_source_documents_are_required_and_digest_bound():
    release, enrollment, vector = material()
    changed = deepcopy(enrollment)
    changed["device_binding"] = "OTHER"
    rejects(release, changed, vector)


def test_two_branch_probe_has_distinct_refs_and_policy_session_packet():
    from deployment.windows_stage9_two_branch_probe import (
        REF_NORMAL,
        REF_RECOVERY,
        policy_auth_packet,
    )

    assert REF_NORMAL != REF_RECOVERY
    packet = policy_auth_packet(0x134, (0x018F0001, 0x018F0001), 0x03000000)
    assert int.from_bytes(packet[:2], "big") == 0x8002
    assert int.from_bytes(packet[6:10], "big") == 0x134
    assert packet.count((0x018F0001).to_bytes(4, "big")) == 2


def test_acceptance_disposable_provenance_is_rejected_for_both_authorities():
    release, enrollment, _ = material()
    release["k_recovery"]["provenance"] = "ACCEPTANCE_DISPOSABLE"
    enrollment["k_psa"]["provenance"] = "ACCEPTANCE_DISPOSABLE"
    enrollment["release_policy_digest"] = canonical_digest(release).hex()
    rejects(release, enrollment)
