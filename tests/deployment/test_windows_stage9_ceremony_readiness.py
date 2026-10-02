from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from deployment.windows_stage9_ceremony_readiness import (
    CURRENT_STATUS,
    EXPECTED_MANIFEST,
    ReadinessError,
    ceremony_entry_gate,
    manifest_digest,
    public_authority_preflight,
    validate_current_status,
    validate_expected_manifest,
)
from deployment.windows_stage9_policy_material import canonical_json_bytes
from deployment.windows_stage9_production_ceremony import (
    build_pdsa_public_bundle,
    build_recovery_public_bundle,
    build_root_anchor_bundle,
)

FIXTURE = Path(__file__).parents[1] / "fixtures/windows_stage9_policy_vector_v1.json"
RUNBOOK = Path(__file__).parents[2] / "docs/windows_stage9_production_root_ceremony.md"


def _records(prefix: str, seeds: tuple[int, int, int]):
    records = []
    for number, seed in enumerate(seeds, 1):
        public = (
            Ed25519PrivateKey.from_private_bytes(bytes([seed]) * 32)
            .public_key()
            .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            .hex()
        )
        records.append(
            {
                "key_id": f"{prefix}-{number}",
                "algorithm": "Ed25519",
                "encoding": "RFC8032_RAW_32_BYTES_LOWER_HEX",
                "public_key_hex": public,
            }
        )
    return records


@pytest.fixture
def authorities(tmp_path: Path):
    root, _ = build_root_anchor_bundle(
        _records("root", (1, 2, 3)), purpose="PRODUCTION", environment="PRODUCTION_CEREMONY"
    )
    pdsa = build_pdsa_public_bundle(_records("pdsa", (4, 5, 6)), threshold=2, purpose="PRODUCTION")
    release = json.loads(FIXTURE.read_text())["release_policy"]
    recovery = build_recovery_public_bundle(
        key_id="PRODUCTION_K_RECOVERY",
        tpmt_public_hex=release["k_recovery"]["public_hex"],
        purpose="PRODUCTION",
        provenance="PRODUCTION_PROVISIONED",
    )
    for name, value in (
        ("root_anchor_bundle", root),
        ("pdsa_public_bundle", pdsa),
        ("recovery_public_bundle", recovery),
    ):
        (tmp_path / f"{name}.json").write_bytes(canonical_json_bytes(value) + b"\n")
    expected = json.loads(EXPECTED_MANIFEST.read_text())
    expected["product_release_root"]["canonical_key_set_sha256"] = root["canonical_key_set_digest"]
    expected["pdsa"]["canonical_key_set_sha256"] = pdsa["canonical_key_set_digest"]
    expected["k_recovery"]["tpmt_public_sha256"] = recovery["tpmt_public_sha256"]
    expected["k_recovery"]["name"] = recovery["derived_name"]
    manifest = tmp_path / "expected.json"
    manifest.write_bytes(canonical_json_bytes(expected))
    return tmp_path, manifest, expected


def test_committed_manifest_is_canonical_public_authority():
    value = json.loads(EXPECTED_MANIFEST.read_text())
    validate_expected_manifest(value)
    assert value["k_recovery"]["name"] == "000b" + value["k_recovery"]["tpmt_public_sha256"]
    assert len(manifest_digest(value)) == 64
    forbidden = {"private", "seed", "password", "pin", "auth_value", "secret"}
    assert not any(word in EXPECTED_MANIFEST.read_text().lower() for word in forbidden)


def test_public_preflight_passes_and_revision_binds_evidence(authorities):
    directory, manifest, expected = authorities
    evidence = public_authority_preflight(
        directory, manifest_path=manifest, revision="a" * 40, timestamp="2026-10-02T00:00:00Z"
    )
    assert evidence["result"] == "PASS"
    assert evidence["git_revision"] == "a" * 40
    assert evidence["expected_manifest_sha256"] == manifest_digest(expected)
    assert set(evidence["actual_public_projection_digests"]) == {
        "product_release_root_sha256",
        "pdsa_sha256",
        "k_recovery_tpmt_public_sha256",
        "k_recovery_name",
    }
    assert evidence["actual_public_authority_profiles"] == {
        "product_release_root": {
            "purpose": "PRODUCTION",
            "environment": "PRODUCTION_CEREMONY",
            "algorithm": "Ed25519",
            "encoding": "RFC8032_RAW_32_BYTES_LOWER_HEX",
            "threshold": 2,
            "key_count": 3,
        },
        "pdsa": {
            "purpose": "PRODUCTION",
            "algorithm": "Ed25519",
            "encoding": "RFC8032_RAW_32_BYTES_LOWER_HEX",
            "threshold": 2,
            "key_count": 3,
        },
    }


@pytest.mark.parametrize(
    ("purpose", "environment"),
    [
        ("TEST_ONLY", "UNIT_TEST_ONLY"),
        ("PRODUCTION", "WRONG_PRODUCTION_ENVIRONMENT"),
    ],
)
def test_root_wrong_purpose_or_environment_fails_with_same_keys(authorities, purpose, environment):
    directory, manifest, expected = authorities
    wrong_root, _ = build_root_anchor_bundle(
        _records("root", (1, 2, 3)), purpose=purpose, environment=environment
    )
    (directory / "root_anchor_bundle.json").write_bytes(canonical_json_bytes(wrong_root) + b"\n")
    expected["product_release_root"]["canonical_key_set_sha256"] = wrong_root[
        "canonical_key_set_digest"
    ]
    manifest.write_bytes(canonical_json_bytes(expected))
    with pytest.raises(ReadinessError, match="ROOT_PROFILE_MISMATCH"):
        public_authority_preflight(directory, manifest_path=manifest, revision="a" * 40)


@pytest.mark.parametrize("threshold", [1, 3])
def test_pdsa_wrong_threshold_fails_with_same_keys_and_digest(authorities, threshold):
    directory, manifest, expected = authorities
    wrong_pdsa = build_pdsa_public_bundle(
        _records("pdsa", (4, 5, 6)), threshold=threshold, purpose="PRODUCTION"
    )
    assert wrong_pdsa["canonical_key_set_digest"] == expected["pdsa"]["canonical_key_set_sha256"]
    (directory / "pdsa_public_bundle.json").write_bytes(canonical_json_bytes(wrong_pdsa) + b"\n")
    with pytest.raises(ReadinessError, match="PDSA_PROFILE_MISMATCH"):
        public_authority_preflight(directory, manifest_path=manifest, revision="a" * 40)


@pytest.mark.parametrize("field", ["root", "pdsa", "recovery_digest", "recovery_name"])
def test_wrong_authority_fails_closed(authorities, field):
    directory, manifest, expected = authorities
    wrong = deepcopy(expected)
    target = {
        "root": ("product_release_root", "canonical_key_set_sha256"),
        "pdsa": ("pdsa", "canonical_key_set_sha256"),
        "recovery_digest": ("k_recovery", "tpmt_public_sha256"),
        "recovery_name": ("k_recovery", "name"),
    }[field]
    wrong[target[0]][target[1]] = ("000b" if field == "recovery_name" else "") + "0" * 64
    if field == "recovery_digest":
        wrong["k_recovery"]["name"] = "000b" + "0" * 64
    manifest.write_bytes(canonical_json_bytes(wrong))
    with pytest.raises(ReadinessError, match="AUTHORITY_MISMATCH|EXPECTED_RECOVERY_NAME"):
        public_authority_preflight(directory, manifest_path=manifest, revision="a" * 40)


def test_missing_malformed_and_unexpected_profile_fail(authorities):
    directory, manifest, _ = authorities
    (directory / "pdsa_public_bundle.json").unlink()
    with pytest.raises(ReadinessError, match="MISSING_OR_MALFORMED"):
        public_authority_preflight(directory, manifest_path=manifest, revision="a" * 40)
    (directory / "pdsa_public_bundle.json").write_text("{")
    with pytest.raises(ReadinessError, match="MISSING_OR_MALFORMED"):
        public_authority_preflight(directory, manifest_path=manifest, revision="a" * 40)
    malformed = json.loads(manifest.read_text())
    malformed["pdsa"]["algorithm"] = "RSA"
    with pytest.raises(ReadinessError, match="EXPECTED_PDSA_PROFILE"):
        validate_expected_manifest(malformed)
    malformed = json.loads(EXPECTED_MANIFEST.read_text())
    malformed["generation"] = "UNREVIEWED_GENERATION"
    with pytest.raises(ReadinessError, match="EXPECTED_MANIFEST_PROFILE"):
        validate_expected_manifest(malformed)
    malformed = json.loads(EXPECTED_MANIFEST.read_text())
    malformed["source_freeze_artifact"] = "elsewhere.md"
    with pytest.raises(ReadinessError, match="EXPECTED_MANIFEST_PROFILE"):
        validate_expected_manifest(malformed)


def test_current_status_and_contradiction_regressions():
    status = json.loads(CURRENT_STATUS.read_text())
    assert validate_current_status(status) == []
    cases = [
        ("production_provisioning_ready", True, "PROVISIONING_BEFORE_CEREMONY"),
        ("windows_0_14", "9/15 DONE", "STALE_WINDOWS_COUNT"),
        ("stage_10_production_lifecycle_live", "LIVE_PASS", "STAGE10_BEFORE_CEREMONY"),
    ]
    for key, value, reason in cases:
        broken = deepcopy(status)
        broken[key] = value
        assert reason in validate_current_status(broken)
    for key, value in (("version", 2), ("stage_10_prerequisite", "UNBLOCKED")):
        broken = deepcopy(status)
        broken[key] = value
        assert validate_current_status(broken)


def _reviewed_repo(tmp_path: Path, manifest: Path, *, include_manifest=True, include_status=True):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    deployment = repo / "deployment"
    deployment.mkdir()
    if include_manifest:
        (deployment / "stage9_expected_public_authorities.json").write_bytes(manifest.read_bytes())
    if include_status:
        (deployment / "stage9_current_status.json").write_bytes(CURRENT_STATUS.read_bytes())
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "test"], cwd=repo, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    return repo, head


def test_entry_gate_checks_canonical_paths_revision_clean_tree_and_authority(authorities, tmp_path):
    directory, manifest, _ = authorities
    repo, head = _reviewed_repo(tmp_path, manifest)
    assert (
        ceremony_entry_gate(repo=repo, reviewed_revision=head, authority_directory=directory) == []
    )
    assert "REVISION_MISMATCH" in ceremony_entry_gate(
        repo=repo, reviewed_revision="0" * 40, authority_directory=directory
    )
    (repo / "untracked").write_text("dirty")
    assert "WORKTREE_NOT_CLEAN" in ceremony_entry_gate(
        repo=repo, reviewed_revision=head, authority_directory=directory
    )


def test_entry_gate_rejects_external_manifest_and_status(authorities, tmp_path):
    directory, manifest, _ = authorities
    repo, head = _reviewed_repo(tmp_path, manifest)
    external_manifest = tmp_path / "fake-manifest.json"
    external_status = tmp_path / "fake-status.json"
    external_manifest.write_bytes(manifest.read_bytes())
    external_status.write_bytes(CURRENT_STATUS.read_bytes())
    assert "NONCANONICAL_MANIFEST_PATH" in ceremony_entry_gate(
        repo=repo,
        reviewed_revision=head,
        authority_directory=directory,
        manifest_path=external_manifest,
    )
    assert "NONCANONICAL_STATUS_PATH" in ceremony_entry_gate(
        repo=repo,
        reviewed_revision=head,
        authority_directory=directory,
        status_path=external_status,
    )


@pytest.mark.parametrize(
    ("relative_path", "reason"),
    [
        ("deployment/stage9_expected_public_authorities.json", "MANIFEST_REVISION_BINDING_FAILED"),
        ("deployment/stage9_current_status.json", "STATUS_REVISION_BINDING_FAILED"),
    ],
)
def test_entry_gate_rejects_working_file_different_from_reviewed_commit(
    authorities, tmp_path, relative_path, reason
):
    directory, manifest, _ = authorities
    repo, head = _reviewed_repo(tmp_path, manifest)
    (repo / relative_path).write_bytes(b"{}\n")
    reasons = ceremony_entry_gate(repo=repo, reviewed_revision=head, authority_directory=directory)
    assert reason in reasons
    assert "WORKTREE_NOT_CLEAN" in reasons


@pytest.mark.parametrize(
    ("include_manifest", "include_status", "reason"),
    [
        (False, True, "MANIFEST_REVISION_BINDING_FAILED"),
        (True, False, "STATUS_REVISION_BINDING_FAILED"),
    ],
)
def test_entry_gate_rejects_file_absent_from_reviewed_commit(
    authorities, tmp_path, include_manifest, include_status, reason
):
    directory, manifest, _ = authorities
    repo, head = _reviewed_repo(
        tmp_path, manifest, include_manifest=include_manifest, include_status=include_status
    )
    reasons = ceremony_entry_gate(repo=repo, reviewed_revision=head, authority_directory=directory)
    assert reason in reasons


def test_runbook_operator_contract():
    text = RUNBOOK.read_text(encoding="utf-8")
    for phrase in (
        "ABORT WITHOUT MUTATION",
        "MUTATING / POINT OF NO RETURN",
        "Product Release Root: **Ed25519, 3 independent holders/keys, threshold 2-of-3**",
        "PDSA: **Ed25519, 3 independent holders/keys, threshold 2-of-3**",
        "separate offline authorities",
        "C:\\Users\\kamil\\Documents\\GitHub\\Dudzian",
        "C:\\CryptoHunter-Production-Authority",
        "READY_FOR_PRODUCTION_CEREMONY",
        "do not delete or overwrite",
        "final planned mutating step",
        "verify-final-package",
    ):
        assert phrase in text
