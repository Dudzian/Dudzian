"""Linux diagnostic transport for independently installed signer processes.

No signing operation exists. A successful exchange establishes the observed
local process identity and carries diagnostics; it cannot prove private control.
"""

from __future__ import annotations

import json
import math
import os
import select
import socket
import stat
import struct
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_FRAME_BYTES = 16 * 1024
EXCHANGE_TIMEOUT_SECONDS = 3.0
_OPERATIONS = frozenset({"DIAGNOSTIC", "QUALIFY"})
_ROLES = frozenset({"root_signer", "history_signer"})
_ACL_NAMES = {"system.posix_acl_access", "system.posix_acl_default"}
_NAMESPACES = ("mnt", "ipc", "net")
# Linux asm-generic/socket.h; Python may predate the kernel's 6.5 constant.
_SO_PEERPIDFD = getattr(socket, "SO_PEERPIDFD", 77)


class ProductionSignerIPCError(RuntimeError):
    """An unavailable or ambiguous diagnostic exchange fails closed."""


@dataclass(frozen=True, slots=True)
class SignerSocketPolicy:
    """Values derived exclusively from the protected installation manifest."""

    role: str
    path: str
    signer_uid: int
    signer_gid: int
    issuer_uid: int
    issuer_gid: int
    installation_sha256: str

    def __post_init__(self) -> None:
        if type(self.path) is not str:
            raise ProductionSignerIPCError("EXACT_INSTALLED_SIGNER_POLICY_REQUIRED")
        path = Path(self.path)
        if (
            type(self.role) is not str
            or self.role not in _ROLES
            or type(self.path) is not str
            or not path.is_absolute()
            or str(path) != self.path
            or ".." in path.parts
            or len(path.parts) < 3
            or any(type(item) is not int or item <= 0 for item in self.identities)
            or self.signer_uid == self.issuer_uid
            or self.signer_gid == self.issuer_gid
            or not _hex_digest(self.installation_sha256)
        ):
            raise ProductionSignerIPCError("EXACT_INSTALLED_SIGNER_POLICY_REQUIRED")

    @property
    def identities(self) -> tuple[int, int, int, int]:
        return self.signer_uid, self.signer_gid, self.issuer_uid, self.issuer_gid


@dataclass(frozen=True, slots=True)
class SignerIPCObservation:
    peer_pid: int
    peer_uid: int
    peer_gid: int
    peer_start_time: int
    peer_namespaces: tuple[tuple[str, int], ...]
    response: dict[str, Any]


def _hex_digest(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _linux() -> None:
    if sys.platform != "linux" or not hasattr(os, "O_PATH"):
        raise ProductionSignerIPCError("LINUX_PIDFD_IDENTITY_REQUIRED")


def _require_self(uid: int, gid: int) -> None:
    if sys.platform != "linux":
        raise ProductionSignerIPCError("LINUX_PIDFD_IDENTITY_REQUIRED")
    _linux()
    if (
        (os.getuid(), os.geteuid(), os.getgid(), os.getegid()) != (uid, uid, gid, gid)
        or os.getresuid() != (uid, uid, uid)
        or os.getresgid() != (gid, gid, gid)
        or not set(os.getgroups()).issubset({gid})
    ):
        raise ProductionSignerIPCError("EXACT_ISOLATED_PROCESS_IDENTITY_REQUIRED")


def _metadata(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_ctime_ns,
    )


def _no_acl(descriptor: int) -> None:
    if sys.platform != "linux":
        raise ProductionSignerIPCError("LINUX_PIDFD_IDENTITY_REQUIRED")
    # O_PATH does not support fgetxattr. procfd follows the pinned descriptor,
    # never the original pathname, including after a directory rename.
    if _ACL_NAMES.intersection(os.listxattr(f"/proc/self/fd/{descriptor}")):
        raise ProductionSignerIPCError("SIGNER_PATH_ACL_FORBIDDEN")


class _PinnedSocketPath:
    def __init__(self, policy: SignerSocketPolicy, *, require_socket: bool = True) -> None:
        self.policy = policy
        self.descriptors: list[int] = []
        self.socket_fd: int | None = None
        self.initial: tuple[int, ...] | None = None
        try:
            self.parent = os.open("/", os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW)
            self.descriptors.append(self.parent)
            parts = Path(policy.path).parts
            for component in parts[1:-1]:
                self._root_directory(self.parent)
                self.parent = os.open(
                    component, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.parent
                )
                self.descriptors.append(self.parent)
            info = os.fstat(self.parent)
            if (
                not stat.S_ISDIR(info.st_mode)
                or info.st_uid != policy.signer_uid
                or info.st_gid != policy.issuer_gid
                or stat.S_IMODE(info.st_mode) != 0o2710
            ):
                raise ProductionSignerIPCError("DEDICATED_SIGNER_DIRECTORY_REQUIRED")
            _no_acl(self.parent)
            self.name = parts[-1]
            self.address = f"/proc/self/fd/{self.parent}/{self.name}"
            if len(os.fsencode(self.address)) >= 108:
                raise ProductionSignerIPCError("SIGNER_SOCKET_PATH_TOO_LONG")
            if require_socket:
                self.pin_socket()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _root_directory(descriptor: int) -> None:
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise ProductionSignerIPCError("PROTECTED_SIGNER_ANCESTRY_REQUIRED")
        _no_acl(descriptor)

    def pin_socket(self) -> None:
        self.socket_fd = os.open(self.name, os.O_PATH | os.O_NOFOLLOW, dir_fd=self.parent)
        self.descriptors.append(self.socket_fd)
        info = os.fstat(self.socket_fd)
        if (
            not stat.S_ISSOCK(info.st_mode)
            or info.st_uid != self.policy.signer_uid
            or info.st_gid != self.policy.issuer_gid
            or stat.S_IMODE(info.st_mode) != 0o660
            or info.st_nlink != 1
        ):
            raise ProductionSignerIPCError("EXACT_SIGNER_SOCKET_REQUIRED")
        _no_acl(self.socket_fd)
        self.initial = _metadata(info)

    def recheck(self) -> None:
        if self.socket_fd is None:
            raise ProductionSignerIPCError("PINNED_SIGNER_SOCKET_REQUIRED")
        info = os.stat(self.name, dir_fd=self.parent, follow_symlinks=False)
        if _metadata(info) != self.initial or _metadata(os.fstat(self.socket_fd)) != self.initial:
            raise ProductionSignerIPCError("SIGNER_SOCKET_CHANGED")
        _no_acl(self.socket_fd)
        # The root-controlled ancestry cannot be renamed by any service; the
        # signer can replace its own final object, detected above.
        for descriptor in self.descriptors[:-2]:
            self._root_directory(descriptor)
        parent = os.fstat(self.parent)
        if (
            parent.st_uid != self.policy.signer_uid
            or parent.st_gid != self.policy.issuer_gid
            or stat.S_IMODE(parent.st_mode) != 0o2710
        ):
            raise ProductionSignerIPCError("SIGNER_DIRECTORY_CHANGED")
        _no_acl(self.parent)

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            os.close(descriptor)
        self.descriptors.clear()

    def __enter__(self) -> _PinnedSocketPath:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _read_proc(directory: int, name: str) -> bytes:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    try:
        data = os.read(descriptor, 65537)
        if not data or len(data) > 65536:
            raise ProductionSignerIPCError("BOUNDED_PROCESS_IDENTITY_REQUIRED")
        return data
    finally:
        os.close(descriptor)


def _process_identity(
    directory: int, uid: int, gid: int
) -> tuple[int, tuple[tuple[str, int], ...]]:
    raw_stat = _read_proc(directory, "stat")
    # comm may contain spaces and parentheses; numeric fields follow its final ')'.
    start_time = int(raw_stat[raw_stat.rfind(b")") + 2 :].split()[19])
    fields = {}
    for line in _read_proc(directory, "status").splitlines():
        key, _, value = line.partition(b":")
        fields[key] = value.split()
    if (
        start_time <= 0
        or tuple(map(int, fields[b"Uid"])) != (uid,) * 4
        or tuple(map(int, fields[b"Gid"])) != (gid,) * 4
        or not set(map(int, fields[b"Groups"])).issubset({gid})
    ):
        raise ProductionSignerIPCError("EXACT_PEER_PROCESS_IDENTITY_REQUIRED")
    try:
        namespaces = tuple(
            (name, os.stat(f"ns/{name}", dir_fd=directory).st_ino) for name in _NAMESPACES
        )
    except PermissionError:
        # Cross-UID namespace symlinks use ptrace access checks on Linux. No
        # CAP_SYS_PTRACE is granted to make an isolation claim. Empty evidence
        # must remain an explicit qualification blocker at the caller.
        namespaces = ()
    return start_time, namespaces


class _PinnedPeer:
    def __init__(self, peer: socket.socket, uid: int, gid: int) -> None:
        if sys.platform != "linux":
            raise ProductionSignerIPCError("LINUX_PIDFD_IDENTITY_REQUIRED")
        self.pidfd: int | None = None
        self.procfd: int | None = None
        self.pid, self.uid, self.gid = struct.unpack(
            "3i", peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        )
        if (self.uid, self.gid) != (uid, gid) or self.pid <= 0 or self.pid == os.getpid():
            raise ProductionSignerIPCError("EXACT_SEPARATE_PEER_PROCESS_REQUIRED")
        try:
            # The kernel socket-bound handle is essential: pidfd_open(pid)
            # after SO_PEERCRED could pin a reused PID while an inherited
            # socket still carries credentials of the deceased originator.
            self.pidfd = peer.getsockopt(socket.SOL_SOCKET, _SO_PEERPIDFD)
            os.set_inheritable(self.pidfd, False)
            fdinfo_directory = os.open("/proc/self/fdinfo", os.O_RDONLY | os.O_DIRECTORY)
            try:
                fdinfo = _read_proc(fdinfo_directory, str(self.pidfd))
            finally:
                os.close(fdinfo_directory)
            pid_lines = [
                line.split()[1:] for line in fdinfo.splitlines() if line.startswith(b"Pid:")
            ]
            if pid_lines != [[str(self.pid).encode()]]:
                raise ProductionSignerIPCError("SOCKET_BOUND_PEER_PIDFD_REQUIRED")
            self.procfd = os.open(f"/proc/{self.pid}", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            self.initial = _process_identity(self.procfd, uid, gid)
            self.recheck()
        except Exception:
            self.close()
            raise

    def recheck(self) -> None:
        if sys.platform != "linux":
            raise ProductionSignerIPCError("LINUX_PIDFD_IDENTITY_REQUIRED")
        if self.pidfd is None or self.procfd is None:
            raise ProductionSignerIPCError("PINNED_PEER_PROCESS_REQUIRED")
        poller = select.poll()
        poller.register(self.pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)
        if poller.poll(0) or _process_identity(self.procfd, self.uid, self.gid) != self.initial:
            raise ProductionSignerIPCError("PEER_PROCESS_CHANGED_OR_EXITED")

    def close(self) -> None:
        for descriptor in (self.procfd, self.pidfd):
            if descriptor is not None:
                os.close(descriptor)
        self.procfd = self.pidfd = None

    def __enter__(self) -> _PinnedPeer:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _remaining(peer: socket.socket, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if not math.isfinite(remaining) or remaining <= 0:
        raise ProductionSignerIPCError("SIGNER_IPC_DEADLINE_EXCEEDED")
    peer.settimeout(remaining)


def _receive_exact(peer: socket.socket, size: int, deadline: float) -> bytes:
    output = bytearray()
    while len(output) < size:
        _remaining(peer, deadline)
        part = peer.recv(size - len(output))
        if not part:
            raise ProductionSignerIPCError("SIGNER_IPC_DISCONNECTED")
        output.extend(part)
    return bytes(output)


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate protocol field")
        result[key] = value
    return result


def _constant(_value: str) -> Any:
    raise ValueError("non-finite JSON forbidden")


def _finite_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("non-finite JSON forbidden")
    return value


def _receive_frame(peer: socket.socket, deadline: float) -> dict[str, Any]:
    size = struct.unpack("!I", _receive_exact(peer, 4, deadline))[0]
    if not 0 < size <= MAX_FRAME_BYTES:
        raise ProductionSignerIPCError("BOUNDED_SIGNER_FRAME_REQUIRED")
    try:
        value = json.loads(
            _receive_exact(peer, size, deadline).decode("utf-8"),
            object_pairs_hook=_object,
            parse_constant=_constant,
            parse_float=_finite_float,
        )
    except (UnicodeError, ValueError, RecursionError):
        raise ProductionSignerIPCError("STRICT_SIGNER_JSON_REQUIRED") from None
    if type(value) is not dict:
        raise ProductionSignerIPCError("SIGNER_PROTOCOL_OBJECT_REQUIRED")
    return value


def _expect_eof(peer: socket.socket, deadline: float) -> None:
    _remaining(peer, deadline)
    if peer.recv(1):
        raise ProductionSignerIPCError("EXACTLY_ONE_SIGNER_FRAME_REQUIRED")


def _send_frame(peer: socket.socket, value: dict[str, Any], deadline: float) -> None:
    try:
        raw = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()
    except (ValueError, TypeError, RecursionError):
        raise ProductionSignerIPCError("STRICT_SIGNER_JSON_REQUIRED") from None
    if not 0 < len(raw) <= MAX_FRAME_BYTES:
        raise ProductionSignerIPCError("BOUNDED_SIGNER_FRAME_REQUIRED")
    frame = struct.pack("!I", len(raw)) + raw
    sent = 0
    while sent < len(frame):
        _remaining(peer, deadline)
        count = peer.send(frame[sent:])
        if count <= 0:
            raise ProductionSignerIPCError("SIGNER_IPC_DISCONNECTED")
        sent += count


def _request(value: dict[str, Any], policy: SignerSocketPolicy) -> None:
    if (
        set(value) != {"version", "operation", "role", "installation_sha256", "nonce"}
        or type(value["version"]) is not int
        or value["version"] != 1
        or type(value["operation"]) is not str
        or value["operation"] not in _OPERATIONS
        or value["role"] != policy.role
        or value["installation_sha256"] != policy.installation_sha256
        or not _hex_digest(value["nonce"])
    ):
        raise ProductionSignerIPCError("EXACT_DIAGNOSTIC_SIGNER_REQUEST_REQUIRED")


def _response(value: dict[str, Any], request: dict[str, Any]) -> None:
    if (
        set(value) != set(request) | {"status", "production_emission", "facts", "blockers"}
        or any(
            type(value[key]) is not type(item) or value[key] != item
            for key, item in request.items()
        )
        or value["status"] != "BLOCKED"
        or value["production_emission"] != "BLOCKED"
        or type(value["facts"]) is not dict
        or type(value["blockers"]) is not list
        or not value["blockers"]
        or any(type(item) is not str or not item or len(item) > 256 for item in value["blockers"])
    ):
        raise ProductionSignerIPCError("EXACT_BLOCKED_DIAGNOSTIC_RESPONSE_REQUIRED")


def qualify_signer_ipc(
    policy: SignerSocketPolicy, operation: str = "DIAGNOSTIC"
) -> SignerIPCObservation:
    """One pinned request, with no reconnect, replay or automatic retry."""
    if sys.platform != "linux":
        raise ProductionSignerIPCError("LINUX_PIDFD_IDENTITY_REQUIRED")
    _require_self(policy.issuer_uid, policy.issuer_gid)
    request = {
        "version": 1,
        "operation": operation,
        "role": policy.role,
        "installation_sha256": policy.installation_sha256,
        "nonce": os.urandom(32).hex(),
    }
    _request(request, policy)
    deadline = time.monotonic() + EXCHANGE_TIMEOUT_SECONDS
    try:
        with (
            _PinnedSocketPath(policy) as path,
            socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer,
        ):
            _remaining(peer, deadline)
            peer.connect(path.address)
            with _PinnedPeer(peer, policy.signer_uid, policy.signer_gid) as identity:
                path.recheck()
                _send_frame(peer, request, deadline)
                peer.shutdown(socket.SHUT_WR)
                response = _receive_frame(peer, deadline)
                _expect_eof(peer, deadline)
                _response(response, request)
                identity.recheck()
                path.recheck()
                _remaining(peer, deadline)
                return SignerIPCObservation(
                    identity.pid,
                    identity.uid,
                    identity.gid,
                    identity.initial[0],
                    identity.initial[1],
                    response,
                )
    except (OSError, ValueError, KeyError, IndexError):
        raise ProductionSignerIPCError("SIGNER_IPC_UNAVAILABLE_OR_CHANGED") from None


def _serve_connection(
    peer: socket.socket,
    policy: SignerSocketPolicy,
    callback: Callable[[str], dict[str, Any]],
    path: _PinnedSocketPath,
) -> None:
    deadline = time.monotonic() + EXCHANGE_TIMEOUT_SECONDS
    with _PinnedPeer(peer, policy.issuer_uid, policy.issuer_gid) as identity:
        request = _receive_frame(peer, deadline)
        _expect_eof(peer, deadline)
        _request(request, policy)
        result = callback(request["operation"])
        if type(result) is not dict or set(result) != {"facts", "blockers"}:
            raise ProductionSignerIPCError("EXACT_PUBLIC_DIAGNOSTIC_RESULT_REQUIRED")
        response = dict(request)
        response.update(
            status="BLOCKED",
            production_emission="BLOCKED",
            facts=result["facts"],
            blockers=result["blockers"],
        )
        _response(response, request)
        identity.recheck()
        path.recheck()
        _send_frame(peer, response, deadline)
        identity.recheck()


def run_diagnostic_server(
    policy: SignerSocketPolicy,
    callback: Callable[[str], dict[str, Any]],
    stop_event: Any = None,
) -> None:
    """Bind as the installed signer; existing/stale endpoints require operator action.

    Never unlinks an existing endpoint or recovers by retrying an exchange. The
    signer-owned setgid directory grants issuer traversal and inherited socket
    group ownership without supplementary issuer-group membership.
    """
    if sys.platform != "linux":
        raise ProductionSignerIPCError("LINUX_PIDFD_IDENTITY_REQUIRED")
    _require_self(policy.signer_uid, policy.signer_gid)
    try:
        with _PinnedSocketPath(policy, require_socket=False) as path:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                old_umask = os.umask(0o117)
                try:
                    listener.bind(path.address)
                finally:
                    os.umask(old_umask)
                path.pin_socket()
                listener.listen(8)
                listener.settimeout(0.25)
                try:
                    while stop_event is None or not stop_event.is_set():
                        path.recheck()
                        try:
                            peer, _address = listener.accept()
                        except TimeoutError:
                            continue
                        with peer:
                            try:
                                _serve_connection(peer, policy, callback, path)
                            except (
                                ProductionSignerIPCError,
                                OSError,
                                ValueError,
                                KeyError,
                                IndexError,
                            ):
                                # No diagnostic response can be trusted after any
                                # malformed, unauthorized, timed-out or lost request.
                                continue
                finally:
                    path.recheck()
                    os.unlink(path.name, dir_fd=path.parent)
    except (OSError, ValueError, KeyError, IndexError):
        raise ProductionSignerIPCError("SIGNER_SERVICE_UNAVAILABLE_OR_CHANGED") from None
