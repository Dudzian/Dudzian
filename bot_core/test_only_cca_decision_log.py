"""Linux TEST-only CCA retention experiment; never a production authority.

There is no private key, source admission, qualified close reader or production
ingress here. A fixture callback may construct one synthetic envelope only
after an immutable intent and an external TEST floor reservation are durable.
The callback is never repeated for a retained intent. A missing/partial decision
therefore remains INDETERMINATE; this deliberately exposes the signing/write
gap instead of signing again. The floor is a separate same-host TEST directory,
not an independently protected production rollback domain.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Callable, Iterator, cast
import uuid

from bot_core.independent_custody_completion_qualification import _bytes, _fields, _json, _text
from bot_core.licensing.canonical import canonical_json_bytes

_ZERO = "0" * 64
_MAX_BINDING = 6 * 1024 * 1024
_MAX_ENTRY = 8 * 1024 * 1024
_MAX_RECORD = 9 * 1024 * 1024
_MAX_ENTRIES = 10000
_MANIFEST = "manifest"
_LOCK = "lock"
_FLOOR = "floor"


class TestOnlyCCALogError(RuntimeError):
    """Unavailable, unsafe or inconsistent TEST retention fails closed."""


class TestOnlyCCALogIndeterminate(TestOnlyCCALogError):
    """Only retained original bytes can resolve an interrupted decision."""


class TestOnlyCCALogConflict(TestOnlyCCALogError):
    """An immutable idempotency binding already belongs to another request."""


def _canonical(value: object) -> bytes:
    return cast(bytes, canonical_json_bytes(value))


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return _identity(info) + (info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _no_acl(descriptor: int) -> None:
    if sys.platform != "linux":
        raise TestOnlyCCALogError("TEST_LINUX_REQUIRED")
    if any(name.startswith("system.posix_acl_") for name in os.listxattr(descriptor)):
        raise TestOnlyCCALogError("TEST_LOG_ACL_FORBIDDEN")


class _Directory:
    """Retain dirfds for every component and detect namespace substitutions."""

    def __init__(self, path: Path, expected: tuple[int, int] | None = None) -> None:
        self.path = path
        self.components: list[tuple[int, int | None, str, tuple[int, ...]]] = []
        self.observed: dict[str, tuple[int, ...]] = {}
        self.fd = -1
        try:
            if (
                sys.platform != "linux"
                or not path.is_absolute()
                or ".." in path.parts
                or len(path.parts) > 128
            ):
                raise TestOnlyCCALogError("TEST_LINUX_EXACT_DIRECTORY_REQUIRED")
            parent = None
            root_owner = os.stat(path.anchor, follow_symlinks=False).st_uid
            for part in path.parts:
                descriptor = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
                )
                info = os.fstat(descriptor)
                self.components.append((descriptor, parent, part, _identity(info)))
                # A sticky temporary root is permitted for disposable tests.
                # Other ancestors must not let unrelated users replace paths.
                if (
                    info.st_mode & 0o022 or info.st_uid not in {root_owner, os.geteuid()}
                ) and not info.st_mode & stat.S_ISVTX:
                    raise TestOnlyCCALogError("TEST_LOG_WRITABLE_ANCESTOR")
                _no_acl(descriptor)
                parent = descriptor
            self.fd = self.components[-1][0]
            info = os.fstat(self.fd)
            if (
                info.st_uid != os.geteuid()
                or info.st_gid != os.getegid()
                or stat.S_IMODE(info.st_mode) != 0o700
                or (expected is not None and (info.st_dev, info.st_ino) != expected)
            ):
                raise TestOnlyCCALogError("TEST_LOG_DIRECTORY_PIN_OR_PERMISSIONS_CHANGED")
            _no_acl(self.fd)
        except Exception:
            self.close()
            raise

    def _recheck_directories(self) -> None:
        for descriptor, parent, name, expected in self.components:
            if (
                _identity(os.fstat(descriptor)) != expected
                or _identity(os.stat(name, dir_fd=parent, follow_symlinks=False)) != expected
            ):
                raise TestOnlyCCALogError("TEST_LOG_DIRECTORY_CHANGED")
            _no_acl(descriptor)

    def recheck(self) -> None:
        """Check every pinned inode at scan, mutation and acknowledgement barriers."""
        self._recheck_directories()
        for name, expected_file in self.observed.items():
            if (
                _file_identity(os.stat(name, dir_fd=self.fd, follow_symlinks=False))
                != expected_file
            ):
                raise TestOnlyCCALogError("TEST_LOG_RETAINED_INODE_CHANGED")

    def close(self) -> None:
        for descriptor, _, _, _ in reversed(self.components):
            os.close(descriptor)
        self.components.clear()

    def read(self, name: str, *, mutable: bool = False, limit: int = _MAX_RECORD) -> bytes:
        if sys.platform != "linux":
            raise TestOnlyCCALogError("TEST_LINUX_REQUIRED")
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.fd)
        try:
            before = os.fstat(descriptor)
            if name in self.observed and _file_identity(before) != self.observed[name]:
                raise TestOnlyCCALogError("TEST_LOG_RETAINED_INODE_CHANGED")
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or before.st_gid != os.getegid()
                or stat.S_IMODE(before.st_mode) != (0o600 if mutable else 0o400)
                or before.st_nlink != 1
                or not 0 <= before.st_size <= limit
            ):
                raise TestOnlyCCALogError("TEST_LOG_FILE_UNSAFE")
            _no_acl(descriptor)
            raw = bytearray()
            while len(raw) <= limit:
                part = os.read(descriptor, min(65536, limit + 1 - len(raw)))
                if not part:
                    break
                raw.extend(part)
            if (
                len(raw) != before.st_size
                or _file_identity(os.fstat(descriptor)) != _file_identity(before)
                or _file_identity(os.stat(name, dir_fd=self.fd, follow_symlinks=False))
                != _file_identity(before)
            ):
                raise TestOnlyCCALogError("TEST_LOG_FILE_CHANGED")
            _no_acl(descriptor)
            self.observed[name] = _file_identity(before)
            # This read has checked its own fd/path/ACL before and after reading.
            # Rechecking all earlier files here makes a scan quadratic. They
            # remain pinned and receive a full check at the next scan/mutation
            # barrier and before the enclosing lock can acknowledge a result.
            self._recheck_directories()
            return bytes(raw)
        finally:
            os.close(descriptor)

    def stage(self, name: str, raw: bytes, *, mutable: bool = False) -> None:
        if sys.platform != "linux":
            raise TestOnlyCCALogError("TEST_LINUX_REQUIRED")
        self.recheck()
        descriptor = os.open(
            name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.fd
        )
        try:
            offset = 0
            while offset < len(raw):
                written = os.write(descriptor, raw[offset:])
                if written <= 0:
                    raise OSError("short TEST log write")
                offset += written
            if not mutable:
                os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
            observed = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            if _file_identity(observed) != _file_identity(os.fstat(descriptor)):
                raise TestOnlyCCALogError("TEST_LOG_STAGE_REPLACED")
            _no_acl(descriptor)
            self.observed[name] = _file_identity(observed)
            self.recheck()
        finally:
            os.close(descriptor)

    def sync(self, name: str, *, mutable: bool = False) -> None:
        self.recheck()
        original = self.read(name, mutable=mutable)
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.fd)
        try:
            if _file_identity(os.fstat(descriptor)) != self.observed[name]:
                raise TestOnlyCCALogError("TEST_LOG_SYNC_INODE_CHANGED")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if self.read(name, mutable=mutable) != original:
            raise TestOnlyCCALogError("TEST_LOG_SYNC_CHANGED")
        os.fsync(self.fd)
        self.recheck()

    def publish(self, stage: str, target: str) -> None:
        """Linux atomic RENAME_NOREPLACE: never silently replace an original."""
        self.recheck()
        rename = ctypes.CDLL(None, use_errno=True).renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        if rename(self.fd, os.fsencode(stage), self.fd, os.fsencode(target), 1) != 0:
            number = ctypes.get_errno()
            raise OSError(number, os.strerror(number))
        expected_stage = self.observed.pop(stage)
        published = os.stat(target, dir_fd=self.fd, follow_symlinks=False)
        # rename changes ctime on Linux; pin all other stage identity fields,
        # then retain the new ctime for every subsequent recheck.
        if _file_identity(published)[:-1] != expected_stage[:-1]:
            raise TestOnlyCCALogError("TEST_LOG_PUBLICATION_INODE_CHANGED")
        self.observed[target] = _file_identity(published)
        os.fsync(self.fd)
        self.recheck()


@dataclass(frozen=True, slots=True)
class TestOnlyCCALogPin:
    """Explicit fixture provisioning output, never a production installation pin."""

    log_directory: str
    floor_directory: str
    installation_id: str
    namespace: str
    log_identity: tuple[int, int]
    floor_identity: tuple[int, int]
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class TestOnlyDecisionPosition:
    namespace: str
    sequence: int
    previous_entry_sha256: str
    entry_id: str

    @property
    def document(self) -> dict[str, object]:
        return {
            "namespace": self.namespace,
            "sequence": self.sequence,
            "previous_entry_sha256": self.previous_entry_sha256,
            "entry_id": self.entry_id,
        }


def _floor_document(
    pin: TestOnlyCCALogPin, sequence: int, digest: str, pending: str | None
) -> dict[str, Any]:
    return {
        "schema": "cryptohunter.TEST-ONLY-cca-floor-v1",
        "installation_id": pin.installation_id,
        "namespace": pin.namespace,
        "sequence": sequence,
        "entry_sha256": digest,
        "pending_intent_sha256": pending,
    }


def provision_test_only_cca_log(
    log_directory: Path, floor_directory: Path, *, installation_id: str, namespace: str
) -> TestOnlyCCALogPin:
    """Initialize two existing empty private TEST directories, explicitly once.

    Runtime opening/lookup never creates missing floors, locks or manifests.
    Same-UID fixtures provide no isolation from malicious same-UID processes.
    """
    _text(installation_id)
    _text(namespace)
    log = _Directory(log_directory)
    floor: _Directory | None = None
    try:
        floor = _Directory(floor_directory)
        if (
            log.path == floor.path
            or log.path in floor.path.parents
            or floor.path in log.path.parents
            or os.listdir(log.fd)
            or os.listdir(floor.fd)
        ):
            raise TestOnlyCCALogError("TEST_DISTINCT_EMPTY_FLOOR_AND_LOG_REQUIRED")
        manifest = _canonical(
            {
                "schema": "cryptohunter.TEST-ONLY-cca-log-v1",
                "installation_id": installation_id,
                "namespace": namespace,
            }
        )
        pin = TestOnlyCCALogPin(
            str(log.path),
            str(floor.path),
            installation_id,
            namespace,
            (os.fstat(log.fd).st_dev, os.fstat(log.fd).st_ino),
            (os.fstat(floor.fd).st_dev, os.fstat(floor.fd).st_ino),
            _sha(manifest),
        )
        log.stage(_MANIFEST, manifest)
        log.stage(_LOCK, b"", mutable=True)
        floor.stage(_FLOOR, _canonical(_floor_document(pin, 0, _ZERO, None)), mutable=True)
        os.fsync(log.fd)
        os.fsync(floor.fd)
        log.recheck()
        floor.recheck()
        return pin
    finally:
        if floor is not None:
            floor.close()
        log.close()


def test_only_cca_immutable_binding(envelope: bytes) -> bytes:
    """Extract a TEST fixture's original binding; no trust or signature grant."""
    from bot_core.independent_cca_decision import parse_independent_cca_decision

    return parse_independent_cca_decision(envelope).immutable_binding_bytes


class TestOnlyCCADecisionLog:
    """flock-serialized Linux retention; never source/CCA qualification."""

    status = "NOT_QUALIFIED"
    production_emission = "BLOCKED"

    def __init__(self, pin: TestOnlyCCALogPin) -> None:
        if type(pin) is not TestOnlyCCALogPin:
            raise TestOnlyCCALogError("EXPLICIT_TEST_LOG_PIN_REQUIRED")
        self.pin = pin

    @contextmanager
    def _locked(self) -> Iterator[tuple[_Directory, _Directory]]:
        if sys.platform != "linux":
            raise TestOnlyCCALogError("TEST_LINUX_REQUIRED")
        import fcntl

        log: _Directory | None = None
        floor: _Directory | None = None
        locked: int | None = None
        try:
            log = _Directory(Path(self.pin.log_directory), self.pin.log_identity)
            floor = _Directory(Path(self.pin.floor_directory), self.pin.floor_identity)
            if _sha(log.read(_MANIFEST)) != self.pin.manifest_sha256:
                raise TestOnlyCCALogError("TEST_LOG_MANIFEST_CHANGED")
            log.read(_LOCK, mutable=True, limit=0)
            locked = os.open(_LOCK, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=log.fd)
            lock_identity = _file_identity(os.fstat(locked))
            if lock_identity != log.observed[_LOCK]:
                raise TestOnlyCCALogError("TEST_LOG_LOCK_REPLACED_BEFORE_FLOCK")
            fcntl.flock(locked, fcntl.LOCK_EX)
            if (
                _file_identity(os.stat(_LOCK, dir_fd=log.fd, follow_symlinks=False))
                != lock_identity
            ):
                raise TestOnlyCCALogError("TEST_LOG_LOCK_REPLACED")
            _no_acl(locked)
            log.recheck()
            floor.recheck()
            yield log, floor
            log.recheck()
            floor.recheck()
            if (
                _file_identity(os.stat(_LOCK, dir_fd=log.fd, follow_symlinks=False))
                != lock_identity
            ):
                raise TestOnlyCCALogError("TEST_LOG_LOCK_REPLACED")
        except OSError as exc:
            raise TestOnlyCCALogIndeterminate("TEST_LOG_IO_OUTCOME_UNKNOWN") from exc
        except (ValueError, TypeError, KeyError) as exc:
            raise TestOnlyCCALogError("TEST_LOG_FORMAT_OR_BINDING_INVALID") from exc
        finally:
            if locked is not None:
                try:
                    fcntl.flock(locked, fcntl.LOCK_UN)
                finally:
                    os.close(locked)
            if floor is not None:
                floor.close()
            if log is not None:
                log.close()

    def _read_floor(self, floor: _Directory) -> dict[str, Any]:
        value = _json(floor.read(_FLOOR, mutable=True, limit=4096), limit=4096)
        _fields(value, set(_floor_document(self.pin, 0, _ZERO, None)))
        sequence, digest, pending = (
            value["sequence"],
            value["entry_sha256"],
            value["pending_intent_sha256"],
        )
        if (
            type(sequence) is not int
            or not 0 <= sequence <= _MAX_ENTRIES
            or type(digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or (sequence == 0) != (digest == _ZERO)
            or (
                pending is not None
                and (type(pending) is not str or re.fullmatch(r"[0-9a-f]{64}", pending) is None)
            )
            or value != _floor_document(self.pin, sequence, digest, pending)
        ):
            raise TestOnlyCCALogError("TEST_LOG_FLOOR_BINDING_INVALID")
        return value

    def _advance_floor(
        self, floor: _Directory, expected: dict[str, Any], value: dict[str, object]
    ) -> None:
        # Only this mutable pointer may be replaced. Its directory is separately
        # pinned and all writers must hold the log lock. An interrupted stage is
        # never silently discarded or interpreted as permission to sign again.
        if self._read_floor(floor) != expected or os.listdir(floor.fd) != [_FLOOR]:
            raise TestOnlyCCALogIndeterminate("TEST_FLOOR_CHANGED_OR_UPDATE_INTERRUPTED")
        floor.stage("floor.stage", _canonical(value), mutable=True)
        if self._read_floor(floor) != expected:
            raise TestOnlyCCALogError("TEST_FLOOR_CHANGED_DURING_UPDATE")
        os.replace("floor.stage", _FLOOR, src_dir_fd=floor.fd, dst_dir_fd=floor.fd)
        staged_identity = floor.observed.pop("floor.stage")
        floor.observed.pop(_FLOOR)
        published = os.stat(_FLOOR, dir_fd=floor.fd, follow_symlinks=False)
        if _file_identity(published)[:-1] != staged_identity[:-1]:
            raise TestOnlyCCALogError("TEST_FLOOR_PUBLICATION_INODE_CHANGED")
        floor.observed[_FLOOR] = _file_identity(published)
        os.fsync(floor.fd)
        floor.recheck()

    def _intent(self, raw: bytes) -> dict[str, Any]:
        value = _fields(
            _json(raw, limit=_MAX_RECORD),
            {"schema", "installation_id", "operation_id", "immutable_binding_b64", "position"},
        )
        _text(value["operation_id"], limit=128)
        position = _fields(
            value["position"], {"namespace", "sequence", "previous_entry_sha256", "entry_id"}
        )
        self._binding(
            value["operation_id"], _bytes(value["immutable_binding_b64"], limit=_MAX_BINDING)
        )
        if (
            value["schema"] != "cryptohunter.TEST-ONLY-cca-intent-v1"
            or value["installation_id"] != self.pin.installation_id
            or position["namespace"] != self.pin.namespace
        ):
            raise TestOnlyCCALogError("TEST_LOG_INTENT_BINDING_INVALID")
        return value

    def _decision(self, raw: bytes, intent: dict[str, Any]) -> None:
        from bot_core.independent_cca_decision import (
            _immutable_binding_from_payload,
            _validated_decision,
        )

        # One full validation supplies both the exact position and the shared
        # schema's immutable binding. Do not parse the same envelope four times.
        _, payload = _validated_decision(raw)
        if (
            payload["decision_log_position"] != intent["position"]
            or payload["decision_id"] != intent["position"]["entry_id"]
            or _immutable_binding_from_payload(payload)
            != _bytes(intent["immutable_binding_b64"], limit=_MAX_BINDING)
        ):
            raise TestOnlyCCALogError("TEST_LOG_DECISION_BINDING_CHANGED")

    def _scan(
        self,
        log: _Directory,
        floor: _Directory,
        request: tuple[str, bytes] | None = None,
    ) -> tuple[list[tuple[dict[str, Any], bytes]], dict[str, Any]]:
        names = set(os.listdir(log.fd))
        conflicts: list[dict[str, Any]] = []
        if len(names) > 3 * _MAX_ENTRIES + 2:
            raise TestOnlyCCALogError("TEST_LOG_RETENTION_LIMIT")
        for name in names:
            if name in {_MANIFEST, _LOCK}:
                continue
            if re.fullmatch(r"[0-9]{20}\.(intent|decision)(\.stage)?", name) is None:
                if re.fullmatch(r"conflict\.[0-9a-f]{64}", name) is None:
                    raise TestOnlyCCALogIndeterminate("TEST_LOG_UNKNOWN_OR_INTERRUPTED_FILE")
                raw = log.read(name)
                conflict = _fields(
                    _json(raw, limit=_MAX_RECORD),
                    {
                        "schema",
                        "installation_id",
                        "operation_id",
                        "accepted_binding_sha256",
                        "rejected_binding_b64",
                    },
                )
                accepted = conflict["accepted_binding_sha256"]
                if (
                    name != "conflict." + _sha(raw)
                    or conflict["schema"] != "cryptohunter.TEST-ONLY-cca-conflict-v1"
                    or conflict["installation_id"] != self.pin.installation_id
                    or type(accepted) is not str
                    or re.fullmatch(r"[0-9a-f]{64}", accepted) is None
                ):
                    raise TestOnlyCCALogError("TEST_LOG_CONFLICT_RECORD_CHANGED")
                self._binding(
                    conflict["operation_id"],
                    _bytes(conflict["rejected_binding_b64"], limit=_MAX_BINDING),
                )
                conflicts.append(conflict)
        trusted = self._read_floor(floor)
        if set(os.listdir(floor.fd)) != {_FLOOR}:
            raise TestOnlyCCALogIndeterminate("TEST_FLOOR_UPDATE_INTERRUPTED")
        result: list[tuple[dict[str, Any], bytes]] = []
        previous = _ZERO
        seen: set[str] = set()
        accepted_bindings: dict[str, str] = {}
        sequence = 1
        while f"{sequence:020d}.intent" in names:
            stem = f"{sequence:020d}"
            intent_raw = log.read(stem + ".intent")
            intent = self._intent(intent_raw)
            position = intent["position"]
            if (
                type(position["sequence"]) is not int
                or position["sequence"] != sequence
                or position["previous_entry_sha256"] != previous
                or intent["operation_id"] in seen
            ):
                raise TestOnlyCCALogError("TEST_LOG_SEQUENCE_OR_PREDECESSOR_CONFLICT")
            seen.add(intent["operation_id"])
            accepted_bindings[intent["operation_id"]] = _sha(
                _bytes(intent["immutable_binding_b64"], limit=_MAX_BINDING)
            )
            decision_name = stem + ".decision"
            if sequence > trusted["sequence"]:
                if (
                    request is not None
                    and intent["operation_id"] == request[0]
                    and _bytes(intent["immutable_binding_b64"], limit=_MAX_BINDING) != request[1]
                    and sequence == trusted["sequence"] + 1
                    and trusted["entry_sha256"] == previous
                    and trusted["pending_intent_sha256"] in {None, _sha(intent_raw)}
                ):
                    self._existing(log, [(intent, b"")], request[0], request[1])
                if (
                    sequence != trusted["sequence"] + 1
                    or trusted["entry_sha256"] != previous
                    or trusted["pending_intent_sha256"] != _sha(intent_raw)
                ):
                    raise TestOnlyCCALogIndeterminate(
                        "TEST_LOG_FLOOR_ROLLBACK_OR_UNRESERVED_INTENT"
                    )
                if decision_name not in names:
                    staged = decision_name + ".stage"
                    if staged not in names:
                        raise TestOnlyCCALogIndeterminate(
                            "TEST_LOG_INTENT_WITHOUT_ORIGINAL_DECISION"
                        )
                    raw = log.read(staged, limit=_MAX_ENTRY)
                    self._decision(raw, intent)
                    log.sync(staged)
                    log.publish(staged, decision_name)
                    names.remove(staged)
                    names.add(decision_name)
                raw = log.read(decision_name, limit=_MAX_ENTRY)
                self._decision(raw, intent)
                log.sync(decision_name)
                next_floor = _floor_document(self.pin, sequence, _sha(raw), None)
                log.recheck()
                self._advance_floor(floor, trusted, next_floor)
                trusted = next_floor
            if decision_name not in names:
                raise TestOnlyCCALogIndeterminate("TEST_LOG_DECISION_MISSING_BELOW_FLOOR")
            raw = log.read(decision_name, limit=_MAX_ENTRY)
            self._decision(raw, intent)
            previous = _sha(raw)
            result.append((intent, raw))
            sequence += 1
        expected = {_MANIFEST, _LOCK} | {name for name in names if name.startswith("conflict.")}
        expected |= {
            f"{index:020d}.{suffix}"
            for index in range(1, sequence)
            for suffix in ("intent", "decision")
        }
        if names != expected or trusted != _floor_document(self.pin, len(result), previous, None):
            raise TestOnlyCCALogIndeterminate("TEST_LOG_FLOOR_MISSING_ROLLED_BACK_OR_GAP")
        if any(
            accepted_bindings.get(conflict["operation_id"]) != conflict["accepted_binding_sha256"]
            for conflict in conflicts
        ):
            raise TestOnlyCCALogError("TEST_LOG_CONFLICT_ACCEPTED_BINDING_CHANGED")
        # A prior process may have died after replacing the floor but before
        # its directory fsync. A visible committed pointer alone is not an ack.
        log.recheck()
        floor.sync(_FLOOR, mutable=True)
        return result, trusted

    def _binding(self, operation_id: str, immutable_binding: bytes) -> None:
        from bot_core.independent_cca_decision import validate_cca_immutable_binding

        try:
            _text(operation_id, limit=128)
            if (
                type(immutable_binding) is not bytes
                or not 0 < len(immutable_binding) <= _MAX_BINDING
            ):
                raise ValueError("bounded original TEST binding bytes required")
            value = validate_cca_immutable_binding(immutable_binding)
        except (ValueError, TypeError, KeyError) as exc:
            raise TestOnlyCCALogError("TEST_LOG_IMMUTABLE_BINDING_INVALID") from exc
        if (
            value["installation"]["installation_id"] != self.pin.installation_id
            or value["close"]["operation_id"] != operation_id
            or value["completion_authority"]["provider_namespace"] != self.pin.namespace
        ):
            raise TestOnlyCCALogError("TEST_LOG_EXACT_OPERATION_BINDING_REQUIRED")

    def _existing(
        self,
        log: _Directory,
        entries: list[tuple[dict[str, Any], bytes]],
        operation_id: str,
        binding: bytes,
    ) -> bytes | None:
        for intent, original in entries:
            if intent["operation_id"] != operation_id:
                continue
            accepted = _bytes(intent["immutable_binding_b64"], limit=_MAX_BINDING)
            if accepted == binding:
                return original
            conflict = _canonical(
                {
                    "schema": "cryptohunter.TEST-ONLY-cca-conflict-v1",
                    "installation_id": self.pin.installation_id,
                    "operation_id": operation_id,
                    "accepted_binding_sha256": _sha(accepted),
                    "rejected_binding_b64": base64.b64encode(binding).decode("ascii"),
                }
            )
            name = "conflict." + _sha(conflict)
            if name not in os.listdir(log.fd):
                log.stage(name + ".stage", conflict)
                log.publish(name + ".stage", name)
            elif log.read(name) != conflict:
                raise TestOnlyCCALogError("TEST_LOG_CONFLICT_RETENTION_CHANGED")
            raise TestOnlyCCALogConflict("TEST_LOG_DURABLE_IMMUTABLE_BINDING_CONFLICT")
        return None

    def exact_lookup(self, operation_id: str, immutable_binding: bytes) -> bytes:
        """Return original bytes, or unknown; absence never authorizes signing."""
        self._binding(operation_id, immutable_binding)
        with self._locked() as (log, floor):
            entries, _ = self._scan(log, floor, (operation_id, immutable_binding))
            existing = self._existing(log, entries, operation_id, immutable_binding)
            if existing is None:
                raise TestOnlyCCALogIndeterminate("TEST_LOG_NOT_FOUND_OUTCOME_UNKNOWN")
            return existing

    def decide(
        self,
        operation_id: str,
        immutable_binding: bytes,
        *,
        create_decision: Callable[[TestOnlyDecisionPosition], bytes],
        fault_injector: Callable[[str], None] | None = None,
    ) -> bytes:
        """Retain a synthetic fixture decision, never a production CCA result.

        The fixture must hold its own source lease and lifecycle serialization
        across this call. No boolean can substitute for future qualified ports.
        A fixture callback cannot be used as production admission or SIGN/BIND.
        """
        self._binding(operation_id, immutable_binding)

        def cut(phase: str) -> None:
            if fault_injector is not None:
                fault_injector(phase)

        with self._locked() as (log, floor):
            entries, trusted = self._scan(log, floor, (operation_id, immutable_binding))
            existing = self._existing(log, entries, operation_id, immutable_binding)
            if existing is not None:
                return existing
            if len(entries) >= _MAX_ENTRIES:
                raise TestOnlyCCALogError("TEST_LOG_RETENTION_LIMIT")
            position = TestOnlyDecisionPosition(
                self.pin.namespace,
                len(entries) + 1,
                trusted["entry_sha256"],
                "test-decision-" + uuid.uuid4().hex,
            )
            intent_raw = _canonical(
                {
                    "schema": "cryptohunter.TEST-ONLY-cca-intent-v1",
                    "installation_id": self.pin.installation_id,
                    "operation_id": operation_id,
                    "immutable_binding_b64": base64.b64encode(immutable_binding).decode("ascii"),
                    "position": position.document,
                }
            )
            stem = f"{position.sequence:020d}"
            cut("before_intent")
            log.stage(stem + ".intent.stage", intent_raw)
            cut("after_intent_write")
            log.publish(stem + ".intent.stage", stem + ".intent")
            cut("after_intent_publication")
            reserved = _floor_document(
                self.pin, trusted["sequence"], trusted["entry_sha256"], _sha(intent_raw)
            )
            log.recheck()
            self._advance_floor(floor, trusted, reserved)
            cut("after_floor_reservation")
            original = create_decision(position)
            self._decision(original, self._intent(intent_raw))
            log.recheck()
            floor.recheck()
            log.stage(stem + ".decision.stage", original)
            cut("after_decision_write")
            log.publish(stem + ".decision.stage", stem + ".decision")
            cut("after_decision_publication")
            log.recheck()
            self._advance_floor(
                floor, reserved, _floor_document(self.pin, position.sequence, _sha(original), None)
            )
            cut("after_durable_decision")
            # Re-read all original bytes and pins before an acknowledged return.
            entries, _ = self._scan(log, floor)
            return entries[-1][1]


__all__ = [
    "TestOnlyCCADecisionLog",
    "TestOnlyCCALogConflict",
    "TestOnlyCCALogError",
    "TestOnlyCCALogIndeterminate",
    "TestOnlyCCALogPin",
    "TestOnlyDecisionPosition",
    "provision_test_only_cca_log",
    "test_only_cca_immutable_binding",
]
