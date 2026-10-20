"""S9-RPI-26: hash proposed systemd unit fragments for diagnosis only.

A digest from an unpinned host is NOT an independently approved install
manifest, unit provenance, executable attestation, or production evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from contextlib import ExitStack

from deployment.stage9_native_linux_cgroup_membership_observation import _run_systemctl
from deployment.stage9_native_linux_process_identity_observation import (
    _PROPOSED_SYSTEMD_UNITS,
    _SYSTEMD_PROPERTIES,
    _parse_unit_hint,
)

_PROPERTIES = (*_SYSTEMD_PROPERTIES, "FragmentPath", "DropInPaths")
_APPROVED_OBSERVATION_DIRS = (
    "/etc/systemd/system",
    "/usr/lib/systemd/system",
    "/lib/systemd/system",
)
_MAX_SYSTEMD_RESPONSE = 8192
_MAX_FRAGMENT_BYTES = 131072
_CODES = frozenset(
    {
        "NATIVE_LINUX_REQUIRED",
        "SYSTEMD_UNAVAILABLE",
        "MALFORMED_SYSTEMD_RESPONSE",
        "UNEXPECTED_SYSTEMD_DROPINS",
        "FRAGMENT_PATH_NOT_ALLOWED",
        "FRAGMENT_PATH_METADATA_NOT_OBSERVABLE",
        "FRAGMENT_NOT_REGULAR_OR_SAFE",
        "FRAGMENT_TOO_LARGE_OR_EMPTY",
        "FRAGMENT_CHANGED_DURING_OBSERVATION",
        "FRAGMENT_READER_INVALID",
        "PROCESS_OR_SYSTEMD_UNAVAILABLE",
        "BACKEND_OBSERVATION_FAILED",
    }
)
_BLOCKERS = (
    "UNIT_HASH_IS_NOT_INDEPENDENTLY_APPROVED_INSTALLATION_PIN",
    "SYSTEMD_HOST_AND_FILESYSTEM_PROVENANCE_NOT_TRUSTED",
    "PROCESS_EXECUTABLE_AND_SOCKET_PEER_NOT_BOUND_TO_FRAGMENT",
    "CROSS_AUTHORITY_EFFECTIVE_ACCESS_NOT_QUALIFIED",
    "SOURCE_CCA_POSTGRESQL_CLOSE_AND_DURABILITY_NOT_QUALIFIED",
    "PRODUCTION_SIGN_BIND_ENROLLMENT_BLOCKED",
)


class _Refusal(ValueError):
    def __init__(self, code: str):
        self.code = code if type(code) is str and code in _CODES else "BACKEND_OBSERVATION_FAILED"
        super().__init__(self.code)


def _read_unit_hint(raw: str, *, unit: str, account: str) -> tuple[int, str]:
    """Parse a closed property list and refuse any drop-in overrides."""
    if (
        type(raw) is not str
        or not raw.isascii()
        or len(raw) > _MAX_SYSTEMD_RESPONSE
        or not raw.endswith("\n")
        or any(ord(c) < 32 and c != "\n" for c in raw)
    ):
        raise _Refusal("MALFORMED_SYSTEMD_RESPONSE")
    fields = {}
    for line in raw[:-1].split("\n"):
        if "=" not in line:
            raise _Refusal("MALFORMED_SYSTEMD_RESPONSE")
        key, value = line.split("=", 1)
        if key not in _PROPERTIES or key in fields:
            raise _Refusal("MALFORMED_SYSTEMD_RESPONSE")
        fields[key] = value
    if set(fields) != set(_PROPERTIES):
        raise _Refusal("MALFORMED_SYSTEMD_RESPONSE")
    if fields["DropInPaths"] != "":
        raise _Refusal("UNEXPECTED_SYSTEMD_DROPINS")
    path = fields["FragmentPath"]
    if type(path) is not str or not any(
        path == f"{directory}/{unit}" for directory in _APPROVED_OBSERVATION_DIRS
    ):
        raise _Refusal("FRAGMENT_PATH_NOT_ALLOWED")
    original = "\n".join(f"{field}={fields[field]}" for field in _SYSTEMD_PROPERTIES)
    try:
        pid = _parse_unit_hint(original + "\n", unit=unit, account=account)
    except ValueError as exc:
        raise _Refusal("MALFORMED_SYSTEMD_RESPONSE") from exc
    return pid, path


def _metadata(item) -> tuple:
    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_gid,
        item.st_nlink,
        item.st_size,
        item.st_ctime_ns,
        item.st_mtime_ns,
    )


def _safe_file_metadata(item) -> bool:
    return (
        stat.S_ISREG(item.st_mode)
        and item.st_uid == 0
        and item.st_gid == 0
        and item.st_nlink == 1
        and not (stat.S_IMODE(item.st_mode) & 0o022)
    )


def _hash_regular_fragment(fd: int, expected) -> dict:
    """Hash the caller-owned fd only after binding it to pre-open metadata."""
    before = os.fstat(fd)
    if _metadata(expected) != _metadata(before):
        raise _Refusal("FRAGMENT_CHANGED_DURING_OBSERVATION")
    if not _safe_file_metadata(before):
        raise _Refusal("FRAGMENT_NOT_REGULAR_OR_SAFE")
    if not 0 < before.st_size <= _MAX_FRAGMENT_BYTES:
        raise _Refusal("FRAGMENT_TOO_LARGE_OR_EMPTY")
    digest = hashlib.sha256()
    size = 0
    while size <= _MAX_FRAGMENT_BYTES:
        chunk = os.read(fd, min(8192, _MAX_FRAGMENT_BYTES + 1 - size))
        if not chunk:
            break
        size += len(chunk)
        if size > _MAX_FRAGMENT_BYTES:
            raise _Refusal("FRAGMENT_TOO_LARGE_OR_EMPTY")
        digest.update(chunk)
    after = os.fstat(fd)
    if size != before.st_size or _metadata(before) != _metadata(after):
        raise _Refusal("FRAGMENT_CHANGED_DURING_OBSERVATION")
    return {"sha256": digest.hexdigest(), "size": size}


def _read_fragment(path: str) -> dict:
    """Pin every path component and bind the single read to checked metadata."""
    if not any(
        path == f"{directory}/{unit}"
        for directory in _APPROVED_OBSERVATION_DIRS
        for _role, unit, _account in _PROPOSED_SYSTEMD_UNITS
    ):
        raise _Refusal("FRAGMENT_PATH_NOT_ALLOWED")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    parts = path.split("/")
    try:
        with ExitStack() as descriptors:
            directories = []
            parent_fd = None
            for name in ("/", *parts[1:-1]):
                relative = {} if parent_fd is None else {"dir_fd": parent_fd}
                before = os.lstat(name, **relative)
                if stat.S_ISLNK(before.st_mode):
                    raise _Refusal("FRAGMENT_CHANGED_DURING_OBSERVATION")
                if (
                    not stat.S_ISDIR(before.st_mode)
                    or before.st_uid != 0
                    or bool(stat.S_IMODE(before.st_mode) & 0o022)
                ):
                    raise _Refusal("FRAGMENT_PATH_METADATA_NOT_OBSERVABLE")
                directory_fd = os.open(name, flags | os.O_DIRECTORY, **relative)
                descriptors.callback(os.close, directory_fd)
                if _metadata(before) != _metadata(os.fstat(directory_fd)):
                    raise _Refusal("FRAGMENT_CHANGED_DURING_OBSERVATION")
                directories.append((parent_fd, name, directory_fd, before))
                parent_fd = directory_fd

            name = parts[-1]
            before = os.lstat(name, dir_fd=parent_fd)
            if stat.S_ISLNK(before.st_mode):
                raise _Refusal("FRAGMENT_CHANGED_DURING_OBSERVATION")
            if not _safe_file_metadata(before):
                raise _Refusal("FRAGMENT_NOT_REGULAR_OR_SAFE")
            fd = os.open(name, flags, dir_fd=parent_fd)
            descriptors.callback(os.close, fd)
            observed = _hash_regular_fragment(fd, before)

            # Validate retained objects and their parent/name bindings, without
            # reopening or rereading the fragment. atime is omitted deliberately:
            # our own read can change it. ctime/mtime still detect observed churn.
            for ancestor_fd, name, directory_fd, expected in directories:
                relative = {} if ancestor_fd is None else {"dir_fd": ancestor_fd}
                if _metadata(expected) != _metadata(os.fstat(directory_fd)) or _metadata(
                    expected
                ) != _metadata(os.lstat(name, **relative)):
                    raise _Refusal("FRAGMENT_CHANGED_DURING_OBSERVATION")
            if _metadata(before) != _metadata(os.fstat(fd)) or _metadata(before) != _metadata(
                os.lstat(parts[-1], dir_fd=parent_fd)
            ):
                raise _Refusal("FRAGMENT_CHANGED_DURING_OBSERVATION")
        return observed
    except OSError as exc:
        # Include failures to open/stat/read/close: identity was not confirmed.
        # Only the fixed code crosses the diagnostic boundary, never OS details.
        raise _Refusal("FRAGMENT_CHANGED_DURING_OBSERVATION") from exc


def _sanitize_fragment(observation) -> dict:
    """Use a fresh builtin record and never serialize arbitrary backend data."""
    if type(observation) is not dict or set(observation) != {"sha256", "size"}:
        raise _Refusal("FRAGMENT_READER_INVALID")
    digest, size = observation["sha256"], observation["size"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
        or type(size) is not int
        or not 0 < size <= _MAX_FRAGMENT_BYTES
    ):
        raise _Refusal("FRAGMENT_READER_INVALID")
    return {"sha256": digest, "size": size}


def inspect_host(*, runner=None, fragment_reader=None) -> dict:
    """Fixed role inventory only; production outcome is always blocked."""
    synthetic = runner is not None or fragment_reader is not None
    result = {
        "schema": "CryptoHunter.Stage9NativeLinuxUnitFingerprintObservationV1",
        "status": "BLOCKED",
        "qualification": "NOT_QUALIFIED",
        "production_emission": "BLOCKED",
        "topology": "PROPOSED_NOT_APPROVED",
        "snapshot": "SYNTHETIC_TEST_INPUT" if synthetic else "UNTRUSTED_LIVE_OBSERVATION",
        "blockers": list(_BLOCKERS),
        "fragment_observations": {},
        "observed_failures": [],
    }
    if sys.platform != "linux":
        result["observed_failures"].append("NATIVE_LINUX_REQUIRED")
        return result
    if runner is None:
        runner = _run_systemctl
    if fragment_reader is None:
        fragment_reader = _read_fragment
    for role, unit, account in _PROPOSED_SYSTEMD_UNITS:
        try:
            command = (
                "/usr/bin/systemctl",
                "--system",
                "show",
                "--no-pager",
                f"--property={','.join(_PROPERTIES)}",
                unit,
            )
            response = runner(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=3,
                env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            )
            if type(response.returncode) is not int:
                raise _Refusal("MALFORMED_SYSTEMD_RESPONSE")
            if response.returncode:
                raise _Refusal("SYSTEMD_UNAVAILABLE")
            pid, path = _read_unit_hint(response.stdout, unit=unit, account=account)
            fingerprint = _sanitize_fragment(fragment_reader(path))
            observed = {
                "observation": "UNPINNED_FRAGMENT_DIGEST_ONLY",
                "pid_hint": pid,
                "fragment_sha256": fingerprint["sha256"],
                "fragment_size": fingerprint["size"],
            }
        except UnicodeDecodeError:
            observed = {"observation": "NOT_QUALIFIED", "issues": ["MALFORMED_SYSTEMD_RESPONSE"]}
        except _Refusal as exc:
            code = getattr(exc, "code", None) if type(exc) is _Refusal else None
            if type(code) is not str or code not in _CODES:
                code = "BACKEND_OBSERVATION_FAILED"
            observed = {"observation": "NOT_QUALIFIED", "issues": [code]}
        except (OSError, TimeoutError):
            observed = {
                "observation": "NOT_QUALIFIED",
                "issues": ["PROCESS_OR_SYSTEMD_UNAVAILABLE"],
            }
        except Exception:
            observed = {"observation": "NOT_QUALIFIED", "issues": ["BACKEND_OBSERVATION_FAILED"]}
        result["fragment_observations"][role] = {"unit": unit, **observed}
        if observed["observation"] == "NOT_QUALIFIED":
            result["observed_failures"].extend(
                f"{role}.fragment: {issue}" for issue in observed["issues"]
            )
    return result


def main() -> int:
    print(json.dumps(inspect_host(), ensure_ascii=True, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
