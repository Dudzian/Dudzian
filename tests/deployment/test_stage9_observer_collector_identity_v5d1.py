"""SYNTHETIC identity tests: TEST keys and mocked files confer no host authority."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from deployment import stage9_observer_collector_identity_v5d1 as identity
from deployment.canonical_json import canonical_json_bytes


def _synthetic_policy() -> tuple[dict, dict, Ed25519PrivateKey]:
    # Ephemeral TEST key is never written or installed as an operator authority.
    key = Ed25519PrivateKey.generate()
    roles = sorted(identity._ROLES)
    policy = {
        "profile": "stage9-observer-controller-policy-v5d1",
        "authority_id": "SYNTHETIC-TEST-ONLY",
        "revision": 1,
        "artifacts": [
            {
                "path": "/etc/cryptohunter/observer-v5d1/domain-requirements.json"
                if role == "domain-requirements"
                else f"/synthetic/{role}",
                "role": role,
                "sha256": hashlib.sha256(b"synthetic").hexdigest(),
            }
            for role in roles
        ],
        "environment_sha256": "0" * 64,
    }
    anchor = {
        "profile": "stage9-observer-controller-anchors-v5d1",
        "authority_id": "SYNTHETIC-TEST-ONLY",
        "public_key_hex": key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex(),
        "approved_revision": 1,
        "approved_policy_sha256": hashlib.sha256(canonical_json_bytes(policy)).hexdigest(),
    }
    return anchor, policy, key


def _signed(anchor: dict, policy: dict, key: Ed25519PrivateKey) -> tuple[bytes, bytes]:
    return canonical_json_bytes(anchor), canonical_json_bytes(
        {
            "policy": policy,
            "signature_hex": key.sign(
                canonical_json_bytes(
                    {"domain": "stage9-observer-controller-policy-signature-v5d1", "policy": policy}
                )
            ).hex(),
        }
    )


def _repin(anchor: dict, policy: dict) -> None:
    anchor["approved_policy_sha256"] = hashlib.sha256(canonical_json_bytes(policy)).hexdigest()


def test_synthetic_signature_validates_only_policy_bytes() -> None:
    anchor, policy, key = _synthetic_policy()
    approved = identity.validate_signed_policy_v5d1(*_signed(anchor, policy, key))
    assert approved.authority_id == "SYNTHETIC-TEST-ONLY"
    assert approved.policy_sha256 == anchor["approved_policy_sha256"]
    assert len(approved.artifacts) == 6
    assert not hasattr(approved, "identity_verified")


def test_synthetic_policy_paths_have_posix_semantics() -> None:
    anchor, policy, key = _synthetic_policy()
    approved = identity.validate_signed_policy_v5d1(*_signed(anchor, policy, key))
    assert tuple(pin.path for pin in approved.artifacts) == tuple(
        artifact["path"] for artifact in policy["artifacts"]
    )
    path = identity._absolute_path("/synthetic/controller-code")
    assert type(path) is PurePosixPath
    assert path.is_absolute()
    assert path.parts == ("/", "synthetic", "controller-code")


@pytest.mark.parametrize(
    "path",
    [
        "synthetic/controller-code",
        "C:/synthetic/controller-code",
        "/synthetic\\controller-code",
        "//synthetic/controller-code",
        "/synthetic/./controller-code",
        "/synthetic/../controller-code",
        "/synthetic//controller-code",
        "/synthetic/controller-code/",
    ],
)
def test_synthetic_policy_rejects_noncanonical_posix_paths(path: str) -> None:
    anchor, policy, key = _synthetic_policy()
    policy["artifacts"][0]["path"] = path
    _repin(anchor, policy)
    with pytest.raises(identity.IdentityRefusalV5d1, match="^IDENTITY_PATH_REJECTED$"):
        identity.validate_signed_policy_v5d1(*_signed(anchor, policy, key))


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("bad-signature", "IDENTITY_POLICY_SIGNATURE_INVALID"),
        ("other-authority", "IDENTITY_POLICY_REJECTED"),
        ("rollback", "IDENTITY_POLICY_REJECTED"),
        ("hash-self-claim", "IDENTITY_POLICY_PIN_MISMATCH"),
        ("missing-role", "IDENTITY_POLICY_CLOSURE_INCOMPLETE"),
        ("duplicate", "IDENTITY_DUPLICATE_ARTIFACT"),
        ("symlink-shaped", "IDENTITY_PATH_REJECTED"),
        ("relative-path", "IDENTITY_PATH_REJECTED"),
        ("caller-manifest", "IDENTITY_REQUIREMENTS_PIN_REJECTED"),
        ("unknown-authority-field", "IDENTITY_ANCHOR_REJECTED"),
    ],
)
def test_synthetic_policy_negative_boundaries(mutation: str, reason: str) -> None:
    anchor, policy, key = _synthetic_policy()
    if mutation == "other-authority":
        policy["authority_id"] = "caller"
    elif mutation == "rollback":
        policy["revision"] = 0
    elif mutation == "hash-self-claim":
        policy["environment_sha256"] = "1" * 64
    elif mutation == "missing-role":
        policy["artifacts"][-1]["role"] = "controller-code"
        _repin(anchor, policy)
    elif mutation == "duplicate":
        policy["artifacts"][-1]["path"] = policy["artifacts"][0]["path"]
        _repin(anchor, policy)
    elif mutation == "symlink-shaped":
        policy["artifacts"][0]["path"] = "/synthetic/../untrusted"
        _repin(anchor, policy)
    elif mutation == "relative-path":
        policy["artifacts"][0]["path"] = "caller/file"
        _repin(anchor, policy)
    elif mutation == "caller-manifest":
        for artifact in policy["artifacts"]:
            if artifact["role"] == "domain-requirements":
                artifact["path"] = "/tmp/caller-manifest.json"
        _repin(anchor, policy)
    elif mutation == "unknown-authority-field":
        anchor["origin"] = "CONTROLLER_OBSERVED"
    anchor_raw, envelope_raw = _signed(anchor, policy, key)
    if mutation == "bad-signature":
        envelope = json.loads(envelope_raw)
        envelope["signature_hex"] = "0" * 128
        envelope_raw = canonical_json_bytes(envelope)
    with pytest.raises(identity.IdentityRefusalV5d1, match=f"^{reason}$"):
        identity.validate_signed_policy_v5d1(anchor_raw, envelope_raw)


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b"[]",
        b'{"profile":1,"profile":2}',
        b'{"value":NaN}',
        b'{"value":1.5}',
        b"[" * 2000,
        b"\xff",
        b"x" * (identity.MAX_POLICY_BYTES + 1),
    ],
)
def test_synthetic_malformed_json_fails_closed(raw: bytes) -> None:
    anchor, policy, key = _synthetic_policy()
    _, envelope = _signed(anchor, policy, key)
    with pytest.raises(identity.IdentityRefusalV5d1):
        identity.validate_signed_policy_v5d1(raw, envelope)


def test_public_entrypoint_has_no_caller_authority_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        identity,
        "read_protected_file_v5d1",
        lambda path: (_ for _ in ()).throw(FileNotFoundError()),
    )
    report = identity.verify_controller_identity_v5d1()
    assert report.identity == "COLLECTOR_IDENTITY_UNVERIFIED"
    assert report.execution == "EXECUTION_NOT_AUTHORIZED"
    assert not report.policy_signature_verified
    assert (
        "IDENTITY_POLICY_UNAVAILABLE" if sys.platform == "linux" else "IDENTITY_HOST_UNSUPPORTED"
    ) in report.reason_codes
    with pytest.raises(TypeError):
        identity.verify_controller_identity_v5d1("caller-authority")


@pytest.mark.skipif(sys.platform == "linux", reason="Native unsupported-host refusal")
def test_native_non_linux_host_refuses_before_policy_or_posix_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_access(*args, **kwargs):
        pytest.fail("Unsupported host must refuse before reading policy or opening descriptors")

    monkeypatch.setattr(identity, "read_protected_file_v5d1", unexpected_access)
    monkeypatch.setattr(identity.os, "open", unexpected_access)
    report = identity.verify_controller_identity_v5d1()
    assert "IDENTITY_HOST_UNSUPPORTED" in report.reason_codes
    assert report.identity == "COLLECTOR_IDENTITY_UNVERIFIED"
    assert report.execution == "EXECUTION_NOT_AUTHORIZED"
    assert not report.policy_signature_verified
    assert report.measurements == ()
    assert report.environment_sha256 is None
    with pytest.raises(identity.IdentityRefusalV5d1, match="^IDENTITY_HOST_UNSUPPORTED$"):
        identity._environment_fingerprint()
    with pytest.raises(identity.IdentityRefusalV5d1, match="^IDENTITY_HOST_UNSUPPORTED$"):
        identity._open_protected(identity.ANCHOR_PATH)
    with pytest.raises(identity.IdentityRefusalV5d1, match="^IDENTITY_HOST_UNSUPPORTED$"):
        identity._read_protected(identity.ANCHOR_PATH, identity.MAX_POLICY_BYTES)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux host identity metadata")
def test_synthetic_full_match_retains_independent_launch_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor, policy, key = _synthetic_policy()
    anchor_raw, envelope = _signed(anchor, policy, key)
    monkeypatch.setattr(
        identity,
        "read_protected_file_v5d1",
        lambda path: anchor_raw if path == identity.ANCHOR_PATH else envelope,
    )
    metadata = os.stat(__file__)
    monkeypatch.setattr(identity, "_read_protected", lambda path, limit: (b"synthetic", metadata))
    monkeypatch.setattr(identity, "_environment_fingerprint", lambda: "0" * 64)
    monkeypatch.setattr(identity, "_open_protected", lambda path: os.open(__file__, os.O_RDONLY))
    monkeypatch.setattr(identity, "_required_artifact_paths", lambda: ({}, ()))
    report = identity.verify_controller_identity_v5d1()
    assert report.policy_signature_verified
    assert len(report.measurements) == 6
    assert "IDENTITY_ARTIFACT_HASH_MISMATCH" not in report.reason_codes
    assert "INDEPENDENT_LAUNCH_ATTESTATION_MISSING" in report.reason_codes
    assert "LOADED_EXECUTION_CLOSURE_UNVERIFIED" in report.reason_codes
    assert report.identity == "COLLECTOR_IDENTITY_UNVERIFIED"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux host identity metadata")
def test_synthetic_policy_change_during_measurement_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor, policy, key = _synthetic_policy()
    anchor_raw, envelope = _signed(anchor, policy, key)
    reads = iter((anchor_raw, envelope, b"changed"))
    monkeypatch.setattr(identity, "read_protected_file_v5d1", lambda path: next(reads))
    monkeypatch.setattr(identity, "_required_artifact_paths", lambda: ({}, ()))
    monkeypatch.setattr(identity, "_environment_fingerprint", lambda: "0" * 64)
    monkeypatch.setattr(identity, "_open_protected", lambda path: os.open(__file__, os.O_RDONLY))
    monkeypatch.setattr(
        identity, "_read_protected", lambda path, limit: (b"wrong", os.stat(__file__))
    )
    report = identity.verify_controller_identity_v5d1()
    assert "IDENTITY_POLICY_CHANGED" in report.reason_codes
    assert "IDENTITY_ARTIFACT_HASH_MISMATCH" in report.reason_codes


def test_synthetic_metadata_and_path_protection(tmp_path: Path) -> None:
    metadata = os.stat(__file__)
    with pytest.raises(identity.IdentityRefusalV5d1, match="IDENTITY_PATH_NOT_PROTECTED"):
        identity._protected_metadata(metadata, directory=False)
    with pytest.raises(identity.IdentityRefusalV5d1, match="IDENTITY_POLICY_LOCATION_REJECTED"):
        identity.read_protected_file_v5d1(tmp_path / "operator-anchors.json")
    with pytest.raises(identity.IdentityRefusalV5d1, match="IDENTITY_LIMIT_REJECTED"):
        identity.read_protected_file_v5d1(identity.ANCHOR_PATH, max_bytes=True)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only protected descriptor traversal")
def test_synthetic_nofollow_rejects_symlink_and_bounded_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only protection metadata is mocked: real temporary descriptor traversal
    # exercises nofollow, size bounds and pathname replacement detection.
    monkeypatch.setattr(identity, "_protected_metadata", lambda metadata, directory: None)
    target = tmp_path / "target"
    target.write_bytes(b"synthetic")
    link = tmp_path / "link"
    link.symlink_to(target)
    assert identity._read_protected(target, 9)[0] == b"synthetic"
    with pytest.raises(OSError):
        identity._read_protected(link, 9)
    with pytest.raises(identity.IdentityRefusalV5d1, match="IDENTITY_ARTIFACT_LIMIT_EXCEEDED"):
        identity._read_protected(target, 8)
    with pytest.raises(OSError):
        identity._read_protected(link / "unused", 9)


def test_synthetic_hardlink_and_writable_metadata_rejected() -> None:
    base = os.stat(__file__)
    # stat_result has no named _replace; construct platform-independent fields.
    values = list(base)
    values[4] = 0
    values[3] = 2
    with pytest.raises(identity.IdentityRefusalV5d1, match="IDENTITY_HARDLINK_REJECTED"):
        identity._protected_metadata(os.stat_result(values), directory=False)
    values[3] = 1
    values[0] |= 0o020
    with pytest.raises(identity.IdentityRefusalV5d1, match="IDENTITY_PATH_NOT_PROTECTED"):
        identity._protected_metadata(os.stat_result(values), directory=False)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux host identity metadata")
def test_synthetic_environment_drift_blocks_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    anchor, policy, key = _synthetic_policy()
    anchor_raw, envelope = _signed(anchor, policy, key)
    fingerprints = iter(("0" * 64, "1" * 64))
    monkeypatch.setattr(identity, "_environment_fingerprint", lambda: next(fingerprints))
    monkeypatch.setattr(
        identity,
        "read_protected_file_v5d1",
        lambda path: anchor_raw if path == identity.ANCHOR_PATH else envelope,
    )
    monkeypatch.setattr(identity, "_required_artifact_paths", lambda: ({}, ()))
    monkeypatch.setattr(
        identity, "_read_protected", lambda path, limit: (b"synthetic", os.stat(__file__))
    )
    monkeypatch.setattr(identity, "_open_protected", lambda path: os.open(__file__, os.O_RDONLY))
    report = identity.verify_controller_identity_v5d1()
    assert "IDENTITY_ENVIRONMENT_CHANGED" in report.reason_codes
    assert report.identity == "COLLECTOR_IDENTITY_UNVERIFIED"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux host identity metadata")
def test_synthetic_path_replacement_between_measurement_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    anchor, policy, key = _synthetic_policy()
    anchor_raw, envelope = _signed(anchor, policy, key)
    changed = tmp_path / "changed"
    changed.write_bytes(b"different-inode")
    monkeypatch.setattr(identity, "_environment_fingerprint", lambda: "0" * 64)
    monkeypatch.setattr(
        identity,
        "read_protected_file_v5d1",
        lambda path: anchor_raw if path == identity.ANCHOR_PATH else envelope,
    )
    monkeypatch.setattr(identity, "_required_artifact_paths", lambda: ({}, ()))
    monkeypatch.setattr(
        identity, "_read_protected", lambda path, limit: (b"synthetic", os.stat(__file__))
    )
    monkeypatch.setattr(identity, "_open_protected", lambda path: os.open(changed, os.O_RDONLY))
    report = identity.verify_controller_identity_v5d1()
    assert "IDENTITY_ARTIFACT_CHANGED" in report.reason_codes
    assert all(item.sha256 == item.expected_sha256 for item in report.measurements)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only protected descriptor traversal")
def test_synthetic_special_file_rejected_before_content_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fifo = tmp_path / "operator-pinned-special-file"
    os.mkfifo(fifo)
    original_open = identity.os.open
    original_check = identity._protected_metadata
    final_open_attempts: list[str] = []

    def observed_open(path, flags, *, dir_fd=None):
        if path == fifo.name:
            final_open_attempts.append(path)
        return original_open(path, flags, dir_fd=dir_fd)

    def synthetic_metadata(metadata, *, directory):
        if not directory:
            original_check(metadata, directory=False)

    monkeypatch.setattr(identity.os, "open", observed_open)
    monkeypatch.setattr(identity, "_protected_metadata", synthetic_metadata)
    with pytest.raises(identity.IdentityRefusalV5d1, match="IDENTITY_PATH_NOT_PROTECTED"):
        identity._open_protected(fifo)
    assert final_open_attempts == []
