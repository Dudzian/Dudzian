"""Operator-controlled identity preflight; never authorizes an execution.

The fixed external anchor pins an Ed25519 key, exact policy digest and revision.
Only a separately provisioned, protected host policy supplies expected hashes.
Measurements of this process cannot establish how its already loaded code was
started: even a valid policy retains the independent launch/closure refusal.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from deployment.canonical_json import canonical_json_bytes

_POLICY_DIRECTORY_POSIX = PurePosixPath("/etc/cryptohunter/observer-v5d1")
POLICY_DIRECTORY = Path(_POLICY_DIRECTORY_POSIX)
ANCHOR_PATH = POLICY_DIRECTORY / "operator-anchors.json"
APPROVED_POLICY_PATH = POLICY_DIRECTORY / "approved-policy.json"
REQUIREMENTS_PATH = POLICY_DIRECTORY / "domain-requirements.json"
MAX_POLICY_BYTES = 256 * 1024
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
_ROLES = {
    "controller-code",
    "interpreter",
    "libvirt-code",
    "libvirt-extension",
    "qemu-tool",
    "domain-requirements",
}
_REQUIRED_CODE = (
    "deployment/canonical_json.py",
    "deployment/stage9_observer_collector_identity_v5d1.py",
    "deployment/stage9_observer_kvm_domain_v5d1.py",
    "deployment/stage9_observer_kvm_image_v5d1.py",
    "deployment/stage9_observer_kvm_preflight_v5d1.py",
    "scripts/stage9_observer_kvm_preflight_v5d1.py",
)


class IdentityRefusalV5d1(ValueError):
    """A fixed diagnostic code, without untrusted file contents."""


@dataclass(frozen=True, slots=True)
class ControllerArtifactPinV5d1:
    path: str
    role: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ApprovedControllerPolicyV5d1:
    authority_id: str
    revision: int
    artifacts: tuple[ControllerArtifactPinV5d1, ...]
    environment_sha256: str
    policy_sha256: str


@dataclass(frozen=True, slots=True)
class IdentityArtifactMeasurementV5d1:
    path: str
    role: str
    sha256: str
    expected_sha256: str
    size: int
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True)
class IdentityReportV5d1:
    policy_sha256: str | None
    policy_signature_verified: bool
    measurements: tuple[IdentityArtifactMeasurementV5d1, ...]
    environment_sha256: str | None
    reason_codes: tuple[str, ...]
    authority_id: str | None = None
    policy_revision: int | None = None
    profile: Literal["stage9-observer-controller-identity-v5d1"] = field(
        default="stage9-observer-controller-identity-v5d1", init=False
    )
    identity: Literal["COLLECTOR_IDENTITY_UNVERIFIED"] = field(
        default="COLLECTOR_IDENTITY_UNVERIFIED", init=False
    )
    execution: Literal["EXECUTION_NOT_AUTHORIZED"] = field(
        default="EXECUTION_NOT_AUTHORIZED", init=False
    )


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise IdentityRefusalV5d1(code)


def _absolute_path(value: Any) -> PurePosixPath:
    """Validate signed Linux path data without consulting the host path flavour."""
    _require(
        type(value) is str
        and 1 < len(value) <= 4096
        and value.startswith("/")
        and not any(ord(char) < 32 or ord(char) > 126 or char == "\\" for char in value)
        and all(part not in {"", ".", ".."} for part in value[1:].split("/")),
        "IDENTITY_PATH_REJECTED",
    )
    return PurePosixPath(value)


def _protected_metadata(metadata: os.stat_result, *, directory: bool) -> None:
    _require(
        (stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode))
        and metadata.st_uid == 0
        and metadata.st_mode & 0o022 == 0,
        "IDENTITY_PATH_NOT_PROTECTED",
    )
    if not directory:
        _require(metadata.st_nlink == 1, "IDENTITY_HARDLINK_REJECTED")


def _metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
    )


def _open_protected(path: Path) -> int:
    """Walk held directory FDs; reject every symlink and writable ancestor."""
    _require(sys.platform == "linux", "IDENTITY_HOST_UNSUPPORTED")
    _absolute_path(str(path))
    directory_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        _protected_metadata(os.fstat(directory_fd), directory=True)
        for component in path.parts[1:-1]:
            next_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
            _protected_metadata(os.fstat(directory_fd), directory=True)
        named_metadata = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        _protected_metadata(named_metadata, directory=False)
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
        try:
            opened_metadata = os.fstat(descriptor)
            _protected_metadata(opened_metadata, directory=False)
            _require(
                _metadata_identity(named_metadata) == _metadata_identity(opened_metadata),
                "IDENTITY_ARTIFACT_CHANGED",
            )
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor
    finally:
        os.close(directory_fd)


def _read_protected(path: Path, max_bytes: int) -> tuple[bytes, os.stat_result]:
    descriptor = _open_protected(path)
    try:
        before = os.fstat(descriptor)
        _require(0 <= before.st_size <= max_bytes, "IDENTITY_ARTIFACT_LIMIT_EXCEEDED")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            _require(total <= max_bytes, "IDENTITY_ARTIFACT_LIMIT_EXCEEDED")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        _require(
            _metadata_identity(before) == _metadata_identity(after) and total == before.st_size,
            "IDENTITY_ARTIFACT_CHANGED",
        )
        # Verify the pathname still designates the held artifact after reading.
        reopened = _open_protected(path)
        try:
            _require(
                _metadata_identity(before) == _metadata_identity(os.fstat(reopened)),
                "IDENTITY_ARTIFACT_CHANGED",
            )
        finally:
            os.close(reopened)
        return b"".join(chunks), before
    finally:
        os.close(descriptor)


def read_protected_file_v5d1(path: Path, *, max_bytes: int = MAX_POLICY_BYTES) -> bytes:
    """Read a direct file in the fixed external operator-policy directory.

    This is a protection check, not approval of arbitrary data in that directory.
    Callers must separately bind the bytes to the signed approved policy.
    """
    _require(
        type(max_bytes) is int and 0 < max_bytes <= MAX_POLICY_BYTES, "IDENTITY_LIMIT_REJECTED"
    )
    _require(path.parent == POLICY_DIRECTORY, "IDENTITY_POLICY_LOCATION_REJECTED")
    return _read_protected(path, max_bytes)[0]


def _json(raw: bytes) -> dict[str, Any]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, "IDENTITY_POLICY_MALFORMED")
            result[key] = value
        return result

    _require(type(raw) is bytes and 0 < len(raw) <= MAX_POLICY_BYTES, "IDENTITY_POLICY_MALFORMED")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_pairs)
        _require(type(value) is dict, "IDENTITY_POLICY_MALFORMED")
        canonical_json_bytes(value)
    except (ValueError, UnicodeError, RecursionError, TypeError) as exc:
        raise IdentityRefusalV5d1("IDENTITY_POLICY_MALFORMED") from exc
    return cast(dict[str, Any], value)


def _hex(value: Any, length: int) -> bool:
    return type(value) is str and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def validate_signed_policy_v5d1(
    anchor_raw: bytes, envelope_raw: bytes
) -> ApprovedControllerPolicyV5d1:
    """Verify policy bytes against supplied anchor bytes; no origin acceptance.

    This pure validator also supports SYNTHETIC tests. Only the zero-argument
    host entry point obtains its anchor from the protected fixed host location.
    """
    anchor = _json(anchor_raw)
    envelope = _json(envelope_raw)
    _require(
        set(anchor)
        == {
            "profile",
            "authority_id",
            "public_key_hex",
            "approved_revision",
            "approved_policy_sha256",
        }
        and anchor["profile"] == "stage9-observer-controller-anchors-v5d1"
        and type(anchor["authority_id"]) is str
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", anchor["authority_id"]) is not None
        and _hex(anchor["public_key_hex"], 64)
        and type(anchor["approved_revision"]) is int
        and 1 <= anchor["approved_revision"] <= 9_007_199_254_740_991
        and _hex(anchor["approved_policy_sha256"], 64),
        "IDENTITY_ANCHOR_REJECTED",
    )
    _require(
        set(envelope) == {"policy", "signature_hex"}
        and type(envelope["policy"]) is dict
        and _hex(envelope["signature_hex"], 128),
        "IDENTITY_POLICY_MALFORMED",
    )
    policy = envelope["policy"]
    _require(
        set(policy) == {"profile", "authority_id", "revision", "artifacts", "environment_sha256"}
        and policy["profile"] == "stage9-observer-controller-policy-v5d1"
        and policy["authority_id"] == anchor["authority_id"]
        and type(policy["revision"]) is int
        and policy["revision"] == anchor["approved_revision"]
        and _hex(policy["environment_sha256"], 64)
        and type(policy["artifacts"]) is list
        and 6 <= len(policy["artifacts"]) <= 1024,
        "IDENTITY_POLICY_REJECTED",
    )
    policy_sha256 = hashlib.sha256(canonical_json_bytes(policy)).hexdigest()
    _require(policy_sha256 == anchor["approved_policy_sha256"], "IDENTITY_POLICY_PIN_MISMATCH")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(anchor["public_key_hex"])).verify(
            bytes.fromhex(envelope["signature_hex"]),
            canonical_json_bytes(
                {"domain": "stage9-observer-controller-policy-signature-v5d1", "policy": policy}
            ),
        )
    except (InvalidSignature, ValueError) as exc:
        raise IdentityRefusalV5d1("IDENTITY_POLICY_SIGNATURE_INVALID") from exc
    artifacts: list[ControllerArtifactPinV5d1] = []
    paths: set[str] = set()
    for artifact in policy["artifacts"]:
        _require(
            type(artifact) is dict
            and set(artifact) == {"path", "role", "sha256"}
            and type(artifact["role"]) is str
            and artifact["role"] in _ROLES
            and _hex(artifact["sha256"], 64),
            "IDENTITY_ARTIFACT_PIN_REJECTED",
        )
        path = str(_absolute_path(artifact["path"]))
        _require(path not in paths, "IDENTITY_DUPLICATE_ARTIFACT")
        paths.add(path)
        artifacts.append(ControllerArtifactPinV5d1(path, artifact["role"], artifact["sha256"]))
    _require({pin.role for pin in artifacts} == _ROLES, "IDENTITY_POLICY_CLOSURE_INCOMPLETE")
    _require(
        [pin.path for pin in artifacts if pin.role == "domain-requirements"]
        == [str(_POLICY_DIRECTORY_POSIX / "domain-requirements.json")],
        "IDENTITY_REQUIREMENTS_PIN_REJECTED",
    )
    return ApprovedControllerPolicyV5d1(
        policy["authority_id"],
        policy["revision"],
        tuple(artifacts),
        policy["environment_sha256"],
        policy_sha256,
    )


def _environment_fingerprint() -> str:
    if sys.platform != "linux":
        raise IdentityRefusalV5d1("IDENTITY_HOST_UNSUPPORTED")
    # Values are not emitted; only the canonical fingerprint enters the report.
    observed = {
        "implementation": platform.python_implementation(),
        "version": sys.version,
        "platform": sys.platform,
        "machine": platform.machine(),
        "uid": os.getuid(),
        "gid": os.getgid(),
        "groups": sorted(os.getgroups()),
        "sys_path": list(sys.path),
        "flags": {
            "isolated": sys.flags.isolated,
            "no_site": sys.flags.no_site,
            "ignore_environment": sys.flags.ignore_environment,
        },
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key
            in {
                "LANG",
                "LC_ALL",
                "TZ",
                "PATH",
                "OPENSSL_CONF",
                "OPENSSL_MODULES",
                "SSL_CERT_FILE",
                "SSL_CERT_DIR",
                "XDG_CONFIG_HOME",
                "HOME",
            }
            or key.startswith(("PYTHON", "LD_", "GLIBC_", "LIBVIRT_", "QEMU_"))
        },
    }
    return hashlib.sha256(canonical_json_bytes(observed)).hexdigest()


def _required_artifact_paths() -> tuple[dict[str, str], tuple[str, ...]]:
    root = Path(__file__).absolute().parent.parent
    required = {str(root / relative): "controller-code" for relative in _REQUIRED_CODE}
    reasons: list[str] = []
    # The kernel, rather than caller-controlled sys.executable, selects this file.
    required[os.readlink("/proc/self/exe")] = "interpreter"
    required[str(REQUIREMENTS_PATH)] = "domain-requirements"
    required["/usr/bin/qemu-img"] = "qemu-tool"
    for module_name, role in (("libvirt", "libvirt-code"), ("libvirtmod", "libvirt-extension")):
        module = sys.modules.get(module_name)
        source = getattr(module, "__file__", None)
        if type(source) is str:
            required[str(_absolute_path(source))] = role
        else:
            reasons.append("LIBVIRT_EXECUTION_CODE_UNOBSERVED")
    # Imported repository modules must be included even if absent from the base list.
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if type(source) is str and source.endswith(".py"):
            candidate = Path(source).absolute()
            if candidate.is_relative_to(root) and candidate.parts[len(root.parts)] in {
                "deployment",
                "scripts",
            }:
                required[str(candidate)] = "controller-code"
    return required, tuple(reasons)


def verify_controller_identity_v5d1() -> IdentityReportV5d1:
    """Measure this host against independently provisioned approval; no writes.

    There is deliberately no caller-selected key, policy, hash, authority flag,
    controller identifier or report that can turn this into authenticated launch
    evidence. Even a fully matching approved disk inventory remains unqualified.
    """
    reasons = [
        "INDEPENDENT_LAUNCH_ATTESTATION_MISSING",
        "LOADED_EXECUTION_CLOSURE_UNVERIFIED",
    ]
    measurements: list[IdentityArtifactMeasurementV5d1] = []
    approved: ApprovedControllerPolicyV5d1 | None = None
    environment_sha256: str | None = None
    try:
        _require(sys.platform == "linux", "IDENTITY_HOST_UNSUPPORTED")
        environment_sha256 = _environment_fingerprint()
        anchor_raw = read_protected_file_v5d1(ANCHOR_PATH)
        policy_raw = read_protected_file_v5d1(APPROVED_POLICY_PATH)
        approved = validate_signed_policy_v5d1(anchor_raw, policy_raw)
        if approved.environment_sha256 != environment_sha256:
            reasons.append("IDENTITY_ENVIRONMENT_MISMATCH")
        if not (sys.flags.isolated and sys.flags.ignore_environment):
            reasons.append("IDENTITY_STARTUP_NOT_ISOLATED")
        if not sys.flags.no_site:
            reasons.append("SYSTEM_SITE_BOOTSTRAP_UNVERIFIED")
        required, missing = _required_artifact_paths()
        reasons.extend(missing)
        pinned = {pin.path: pin.role for pin in approved.artifacts}
        if any(pinned.get(path) != role for path, role in required.items()):
            reasons.append("IDENTITY_POLICY_CLOSURE_INCOMPLETE")
        for pin in approved.artifacts:
            try:
                data, metadata = _read_protected(Path(pin.path), MAX_ARTIFACT_BYTES)
                actual_sha256 = hashlib.sha256(data).hexdigest()
                measurements.append(
                    IdentityArtifactMeasurementV5d1(
                        pin.path,
                        pin.role,
                        actual_sha256,
                        pin.sha256,
                        metadata.st_size,
                        metadata.st_dev,
                        metadata.st_ino,
                        metadata.st_mtime_ns,
                        metadata.st_ctime_ns,
                    )
                )
                if actual_sha256 != pin.sha256:
                    reasons.append("IDENTITY_ARTIFACT_HASH_MISMATCH")
            except IdentityRefusalV5d1 as exc:
                reasons.append(str(exc))
            except OSError:
                reasons.append("IDENTITY_ARTIFACT_UNREADABLE")
        for measurement in measurements:
            try:
                descriptor = _open_protected(Path(measurement.path))
                try:
                    current = os.fstat(descriptor)
                    _require(
                        (
                            current.st_dev,
                            current.st_ino,
                            current.st_size,
                            current.st_mtime_ns,
                            current.st_ctime_ns,
                        )
                        == (
                            measurement.device,
                            measurement.inode,
                            measurement.size,
                            measurement.mtime_ns,
                            measurement.ctime_ns,
                        ),
                        "IDENTITY_ARTIFACT_CHANGED",
                    )
                finally:
                    os.close(descriptor)
            except IdentityRefusalV5d1 as exc:
                reasons.append(str(exc))
            except OSError:
                reasons.append("IDENTITY_ARTIFACT_UNREADABLE")
        _require(
            _environment_fingerprint() == environment_sha256,
            "IDENTITY_ENVIRONMENT_CHANGED",
        )
        # Re-read fixed policy/anchor to detect changes during artifact collection.
        _require(
            read_protected_file_v5d1(ANCHOR_PATH) == anchor_raw
            and read_protected_file_v5d1(APPROVED_POLICY_PATH) == policy_raw,
            "IDENTITY_POLICY_CHANGED",
        )
    except IdentityRefusalV5d1 as exc:
        reasons.append(str(exc))
    except OSError:
        reasons.append("IDENTITY_POLICY_UNAVAILABLE")
    return IdentityReportV5d1(
        approved.policy_sha256 if approved else None,
        approved is not None,
        tuple(measurements),
        environment_sha256,
        tuple(dict.fromkeys(reasons)),
        approved.authority_id if approved else None,
        approved.revision if approved else None,
    )
