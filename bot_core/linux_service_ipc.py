"""Odczytowy exchange kwalifikacyjny rzeczywistego verifiera Linux.

Wyłącznie root może odpytać tę gałąź. Nie dostaje dostępu do przygotowania
kandydata, podpisywania, SQL ani administracji. Obserwacja jest częściowa:
nie zastępuje połączenia działającego runtime ani trwałej ochrony replay.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import os
from pathlib import Path
import re
import select
import socket
import stat
import struct
import sys
import time
from typing import Any, cast

from bot_core.production_signer_ipc import (
    EXCHANGE_TIMEOUT_SECONDS,
    ProductionSignerIPCError,
    _SO_PEERPIDFD,
    _expect_eof,
    _read_proc,
    _receive_frame,
    _send_frame,
)

POLICY_PATH = "/etc/cryptohunter/linux-qualification.json"
READONLY_OPERATION = "READONLY_QUALIFY"
PARTIAL_BLOCKERS = (
    "ACTUAL_RUNTIME_CLIENT_EXCHANGE_UNPROVEN",
    "RUNTIME_UNAUTHORIZED_IDENTITY_DENIAL_UNPROVEN",
    "DURABLE_CHALLENGE_REPLAY_PROTECTION_UNAVAILABLE",
)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_REQUEST_FIELDS = {"version", "operation", "installation_id", "challenge_id", "nonce"}
_ACL_NAMES = {"system.posix_acl_access", "system.posix_acl_default"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductionSignerIPCError(message)


def _digest(value: object) -> bool:
    return type(value) is str and _DIGEST.fullmatch(value) is not None


def _ticks(raw: bytes, pid: int) -> int:
    _require(raw.startswith(str(pid).encode() + b" ("), "QUALIFIER_PROCESS_PID_CHANGED")
    fields = raw[raw.rfind(b")") + 2 :].split()
    _require(
        len(fields) >= 20 and fields[0] not in {b"Z", b"X"} and fields[19].isdigit(),
        "QUALIFIER_PROCESS_LIFETIME_REQUIRED",
    )
    result = int(fields[19])
    _require(result > 0, "QUALIFIER_PROCESS_LIFETIME_REQUIRED")
    return result


def capture_process_identity(pid: int, *, allow_missing_executable: bool = False) -> dict[str, Any]:
    """Czytaj /proc przez uchwyt procesu, przed i po sprawdzając czas startu.

    Brak ptrace dla obcego UID nie jest obchodzony. Jawne None dla programu
    pozostaje blockerem, a nie potwierdzeniem tożsamości programu.
    """
    _require(sys.platform == "linux", "LINUX_PIDFD_IDENTITY_REQUIRED")
    _require(type(pid) is int and pid > 0, "QUALIFIER_PROCESS_PID_REQUIRED")
    directory = os.open(f"/proc/{pid}", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        first = _ticks(_read_proc(directory, "stat"), pid)
        fields = {}
        for line in _read_proc(directory, "status").splitlines():
            key, _, value = line.partition(b":")
            _require(key not in fields, "QUALIFIER_PROCESS_STATUS_AMBIGUOUS")
            fields[key] = value.split()
        uids = tuple(map(int, fields[b"Uid"]))
        gids = tuple(map(int, fields[b"Gid"]))
        _require(
            len(uids) == len(gids) == 4 and len(set(uids)) == len(set(gids)) == 1,
            "QUALIFIER_PROCESS_CREDENTIALS_AMBIGUOUS",
        )
        cgroups = _read_proc(directory, "cgroup").decode("utf-8").splitlines()
        _require(
            len(cgroups) == 1 and cgroups[0].startswith("0::/"),
            "QUALIFIER_PROCESS_CGROUP_V2_REQUIRED",
        )
        cgroup = cgroups[0][3:]
        executable: str | None = None
        try:
            # /proc/<pid>/exe jest jawnym jądrowym uchwytem programu.
            descriptor = os.open("exe", os.O_RDONLY, dir_fd=directory)
        except PermissionError:
            if not allow_missing_executable:
                raise
        else:
            try:
                before_executable = os.fstat(descriptor)
                _require(stat.S_ISREG(before_executable.st_mode), "QUALIFIER_PROGRAM_REQUIRED")
                hasher = hashlib.sha256()
                total = 0
                while chunk := os.read(descriptor, 1024 * 1024):
                    total += len(chunk)
                    _require(total <= 512 * 1024 * 1024, "QUALIFIER_PROGRAM_TOO_LARGE")
                    hasher.update(chunk)
                executable = hasher.hexdigest()
                after_executable = os.fstat(descriptor)
                linked_executable = os.stat("exe", dir_fd=directory)
                metadata = lambda info: (
                    info.st_dev,
                    info.st_ino,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                )
                _require(
                    metadata(before_executable)
                    == metadata(after_executable)
                    == metadata(linked_executable),
                    "QUALIFIER_PROGRAM_CHANGED",
                )
            finally:
                os.close(descriptor)
        _require(
            _ticks(_read_proc(directory, "stat"), pid) == first,
            "QUALIFIER_PROCESS_CHANGED",
        )
        return {
            "pid": pid,
            "uid": uids[0],
            "gid": gids[0],
            "groups": sorted(map(int, fields[b"Groups"])),
            "start_time_ticks": first,
            "exe_sha256": executable,
            "cgroup": cgroup,
        }
    finally:
        os.close(directory)


class PinnedReadOnlyPeer:
    """SO_PEERPIDFD wiąże obserwację z faktycznym partnerem danego socketu."""

    def __init__(self, peer: socket.socket, *, uid: int, gid: int) -> None:
        self.pidfd: int | None = None
        self.credentials: dict[str, int]
        self.process: dict[str, Any]
        if sys.platform != "linux":
            raise ProductionSignerIPCError("LINUX_PIDFD_IDENTITY_REQUIRED")
        _require(
            peer.family == socket.AF_UNIX and peer.type == socket.SOCK_STREAM,
            "QUALIFIER_AF_UNIX_STREAM_REQUIRED",
        )
        pid, observed_uid, observed_gid = struct.unpack(
            "3i", peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        )
        _require(
            (observed_uid, observed_gid) == (uid, gid) and pid > 0 and pid != os.getpid(),
            "QUALIFIER_KERNEL_CLIENT_DENIED",
        )
        self.credentials = {"pid": pid, "uid": observed_uid, "gid": observed_gid}
        try:
            self.pidfd = peer.getsockopt(socket.SOL_SOCKET, _SO_PEERPIDFD)
            os.set_inheritable(self.pidfd, False)
            directory = os.open("/proc/self/fdinfo", os.O_RDONLY | os.O_DIRECTORY)
            try:
                fdinfo = _read_proc(directory, str(self.pidfd))
            finally:
                os.close(directory)
            _require(
                [line.split()[1:] for line in fdinfo.splitlines() if line.startswith(b"Pid:")]
                == [[str(pid).encode()]],
                "SOCKET_BOUND_PEER_PIDFD_REQUIRED",
            )
            self.process = capture_process_identity(pid, allow_missing_executable=True)
            _require(
                all(self.process[key] == value for key, value in self.credentials.items()),
                "QUALIFIER_PROCESS_CREDENTIALS_CHANGED",
            )
            self.recheck()
        except Exception:
            self.close()
            raise

    def recheck(self) -> None:
        if sys.platform != "linux":
            raise ProductionSignerIPCError("LINUX_PIDFD_IDENTITY_REQUIRED")
        if self.pidfd is None:
            raise ProductionSignerIPCError("PINNED_PEER_PROCESS_REQUIRED")
        poller = select.poll()
        poller.register(self.pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)
        _require(not poller.poll(0), "QUALIFIER_PEER_EXITED")
        _require(
            capture_process_identity(self.credentials["pid"], allow_missing_executable=True)
            == self.process,
            "QUALIFIER_PEER_CHANGED",
        )
        _require(not poller.poll(0), "QUALIFIER_PEER_EXITED")

    def close(self) -> None:
        if self.pidfd is not None:
            os.close(self.pidfd)
            self.pidfd = None

    def __enter__(self) -> PinnedReadOnlyPeer:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _validate_request(request: object, installation_id: str) -> None:
    _require(
        type(request) is dict
        and set(request) == _REQUEST_FIELDS
        and type(request["version"]) is int
        and request["version"] == 1
        and request["operation"] == READONLY_OPERATION
        and request["installation_id"] == installation_id
        and all(_digest(request[key]) for key in ("installation_id", "challenge_id", "nonce")),
        "EXACT_READONLY_QUALIFICATION_REQUEST_REQUIRED",
    )


class ReadOnlyVerifierQualifier:
    """Stała gałąź odczytowa; żądanie nigdy nie wybiera funkcji callback."""

    def __init__(
        self,
        installation_id: str,
        postgres_observer: Callable[[str, str], dict[str, Any]] | None = None,
    ) -> None:
        _require(_digest(installation_id), "QUALIFIER_INSTALLATION_BINDING_REQUIRED")
        self.installation_id = installation_id
        self.postgres_observer = postgres_observer
        self._used: set[tuple[str, str]] = set()
        self._self = capture_process_identity(os.getpid())

    def serve_connection(self, peer: socket.socket) -> None:
        """Root ma jedynie odczyt; inne UID/GID odrzuca mechanizm SO_PEERCRED."""
        deadline = time.monotonic() + EXCHANGE_TIMEOUT_SECONDS
        with PinnedReadOnlyPeer(peer, uid=0, gid=0) as client:
            request = _receive_frame(peer, deadline)
            _expect_eof(peer, deadline)
            _validate_request(request, self.installation_id)
            replay_key = (request["challenge_id"], request["nonce"])
            _require(replay_key not in self._used, "QUALIFIER_REQUEST_REPLAYED")
            # Nie usuwamy wpisów i nie przyjmujemy nowego żądania po wyczerpaniu
            # limitu. Restart nie daje trwałej ochrony i pozostaje blockerem.
            _require(len(self._used) < 4096, "QUALIFIER_REPLAY_WINDOW_EXHAUSTED")
            self._used.add(replay_key)
            client.recheck()
            _require(
                capture_process_identity(os.getpid()) == self._self,
                "QUALIFIER_SERVICE_PROCESS_CHANGED",
            )
            postgresql = None
            blockers = list(PARTIAL_BLOCKERS)
            if client.process["exe_sha256"] is None:
                blockers.append("SERVER_CLIENT_EXECUTABLE_OBSERVATION_UNAVAILABLE")
            if self.postgres_observer is None:
                blockers.append("VERIFIER_POSTGRESQL_OBSERVER_UNAVAILABLE")
            else:
                postgresql = self.postgres_observer(request["challenge_id"], self.installation_id)
                _require(type(postgresql) is dict, "QUALIFIER_POSTGRESQL_OBSERVATION_REQUIRED")
            response = {
                **request,
                "status": "PARTIAL_VERIFIER_OBSERVATION",
                "production_qualification": "BLOCKED",
                "server_process": self._self,
                "server_observed_client": {
                    "peer_credentials": client.credentials,
                    "process": client.process,
                },
                "socket_bound_pidfd": True,
                "postgresql": postgresql,
                "blockers": blockers,
            }
            client.recheck()
            _require(
                capture_process_identity(os.getpid()) == self._self,
                "QUALIFIER_SERVICE_PROCESS_CHANGED",
            )
            _send_frame(peer, response, deadline)
            client.recheck()


def load_installed_qualification_policy() -> str | None:
    """Opcjonalny stały plik root; brak pliku nie rozszerza starego protokołu.

    installation_id jest niezależnym identyfikatorem, nie hashem manifestu
    zawierającego hash tego samego pliku. Producent dowodów osobno wiąże jego
    dokładne bajty z zatwierdzonym profilem instalacji.
    """
    if sys.platform != "linux":
        raise ProductionSignerIPCError("LINUX_PIDFD_IDENTITY_REQUIRED")
    descriptors = []
    try:
        directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(directory)
        parts = Path(POLICY_PATH).parts
        for component in parts[1:-1]:
            info = os.fstat(directory)
            _require(
                info.st_uid == info.st_gid == 0 and not info.st_mode & 0o022,
                "QUALIFIER_POLICY_ANCESTRY_UNPROTECTED",
            )
            try:
                directory = os.open(
                    component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
                )
            except FileNotFoundError:
                return None
            descriptors.append(directory)
        info = os.fstat(directory)
        _require(
            info.st_uid == info.st_gid == 0 and not info.st_mode & 0o022,
            "QUALIFIER_POLICY_ANCESTRY_UNPROTECTED",
        )
        try:
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        except FileNotFoundError:
            return None
        descriptors.append(descriptor)
        info = os.fstat(descriptor)
        _require(
            stat.S_ISREG(info.st_mode)
            and info.st_uid == info.st_gid == 0
            and stat.S_IMODE(info.st_mode) == 0o644
            and info.st_nlink == 1
            and info.st_size <= 4096,
            "QUALIFIER_POLICY_ROOT_OWNERSHIP_REQUIRED",
        )
        for item in descriptors:
            _require(
                not _ACL_NAMES.intersection(os.listxattr(item)), "QUALIFIER_POLICY_ACL_FORBIDDEN"
            )
        raw = os.read(descriptor, 4097)
        after = os.fstat(descriptor)
        _require(
            len(raw) <= 4096
            and (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "QUALIFIER_POLICY_CHANGED",
        )
        # Ten sam parser protokołu odrzuca powtórzone pola i wartości niefinitywne.
        from bot_core.production_signer_ipc import _constant, _object

        value = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)
        _require(
            type(value) is dict
            and set(value) == {"schema_version", "installation_id"}
            and type(value["schema_version"]) is int
            and value["schema_version"] == 1
            and _digest(value["installation_id"]),
            "QUALIFIER_POLICY_EXACT_FIELDS_REQUIRED",
        )
        return cast(str, value["installation_id"])
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
