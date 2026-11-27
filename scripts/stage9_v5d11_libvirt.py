"""Collect bounded, read-only libvirt observations for the V5d11 qualification."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
import re
import socket
import stat
import struct
import subprocess
import time
from typing import Any
import uuid

SOCKETS = {
    "/run/libvirt/libvirt-sock-ro": "libvirtd",
    "/run/libvirt/virtqemud-sock-ro": "virtqemud",
}
CLIENT = Path("/etc/libvirt/libvirt.conf")
CLIENT_BYTES = b'uri_default = "qemu:///system"\n'
COMMAND_ENV = {"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"}
PROC = Path("/proc")
CONFIG = Path("/etc/libvirt")
CONTROL_LIMIT = 64 * 1024
FILE_LIMIT = 64 * 1024 * 1024
TIMEOUT = 5

# Linux include/uapi/linux/{sock_diag,unix_diag,netlink}.h
NL_HEADER = struct.Struct("=IHHII")
UNIX_MESSAGE = struct.Struct("=BBBBIII")
AF_UNIX = 1
NETLINK_SOCK_DIAG = 4
SOCK_DIAG_BY_FAMILY = 20
UNIX_DIAG_NAME = 0
UNIX_DIAG_VFS = 1
UNIX_DIAG_PEER = 2
TCP_ESTABLISHED = 1
TCP_LISTEN = 10

# libvirt 10.0 remote_protocol.x and virnetprotocol.x. The only requests
# provide a read-only endpoint witness without an operation that can mutate.
REMOTE_PROGRAM = 0x20008086
REMOTE_VERSION = 1
AUTH_LIST = 66
CONNECT_OPEN = 1
LOOKUP_BY_UUID = 24
GET_XML_DESC = 14
SECURE_XML = 1
OPERATION_DENIED = 29
RPC_HEADER = struct.Struct("!6I")
RPC_LENGTH = struct.Struct("!I")
TRACE_MAGIC = b"STAGE9-V5D11-LIBVIRT-RPC\0"
TRACE_METADATA_LIMIT = 32 * 1024
TRACE_LIMIT = len(TRACE_MAGIC) + RPC_LENGTH.size + TRACE_METADATA_LIMIT + 8 * CONTROL_LIMIT
SESSION_FIELDS = {"spec_sha256", "authorization_sha256", "nonce"}
MAX_DIAG_DATAGRAMS = 2048
DiagRow = tuple[int, int, str, int, tuple[int, int] | None]


class Refusal(ValueError):
    """The socket, service, or protocol evidence lacks required identity."""


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise Refusal(reason)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_bounded(path: Path, limit: int = CONTROL_LIMIT) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit, "LIBVIRT_FILE_LIMIT")
    return raw


def stat_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_nlink,
        info.st_uid,
        info.st_gid,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def protected_file(path: Path, anchor: Path = Path("/")) -> bytes:
    """Open an exact regular path from a held root; never follow file links."""
    flags = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    nonblock = getattr(os, "O_NONBLOCK", None)
    require(
        type(flags) is int
        and type(directory) is int
        and type(nonblock) is int
        and path.is_absolute()
        and ".." not in path.parts
        and bool(path.name),
        "LIBVIRT_FILE_PATH",
    )
    assert flags is not None and directory is not None and nonblock is not None
    descriptor = os.open(anchor, os.O_RDONLY | directory)
    try:
        root = os.fstat(descriptor)
        require(root.st_uid == 0 and root.st_mode & 0o022 == 0, "LIBVIRT_FILE_ANCESTRY")
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | directory | flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            metadata = os.fstat(descriptor)
            require(metadata.st_uid == 0 and metadata.st_mode & 0o022 == 0, "LIBVIRT_FILE_ANCESTRY")
        fd = os.open(path.name, os.O_RDONLY | flags | nonblock, dir_fd=descriptor)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            require(
                stat.S_ISREG(info.st_mode)
                and info.st_uid == 0
                and info.st_nlink == 1
                and info.st_mode & 0o022 == 0
                and 0 < info.st_size <= FILE_LIMIT,
                "LIBVIRT_FILE_METADATA",
            )
            raw = stream.read(info.st_size + 1)
            after = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
            require(
                len(raw) == info.st_size
                and stat_identity(info)
                == stat_identity(after)
                == stat_identity(os.fstat(stream.fileno())),
                "LIBVIRT_FILE_CHANGED",
            )
        return raw
    finally:
        os.close(descriptor)


def socket_identity(path: Path) -> list[int]:
    for ancestor in path.parents:
        parent = ancestor.lstat()
        require(
            stat.S_ISDIR(parent.st_mode) and parent.st_uid == 0 and parent.st_mode & 0o022 == 0,
            "LIBVIRT_SOCKET_ANCESTRY",
        )
    info = path.lstat()
    require(
        stat.S_ISSOCK(info.st_mode) and info.st_uid == 0 and info.st_ino > 0,
        "LIBVIRT_SOCKET_METADATA",
    )
    return [info.st_dev, info.st_ino, info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)]


def peer_credentials(connection: socket.socket) -> list[int]:
    option = getattr(socket, "SO_PEERCRED", None)
    require(type(option) is int, "LIBVIRT_PEERCRED_UNAVAILABLE")
    assert option is not None
    raw = connection.getsockopt(socket.SOL_SOCKET, option, 12)
    pid, uid, gid = struct.unpack("=iII", raw)
    require(pid >= 0 and uid == 0, "LIBVIRT_PEER_REJECTED")
    return [pid, uid, gid]


def proc_start(pid: int) -> int:
    _, separator, tail = read_bounded(PROC / str(pid) / "stat").rpartition(b") ")
    values = tail.split()
    require(
        bool(separator) and len(values) > 19 and values[19].isdigit(),
        "LIBVIRT_PROCESS_STAT_REJECTED",
    )
    return int(values[19])


def descriptor_sockets(pid: int) -> set[int]:
    directory = PROC / str(pid) / "fd"
    identities = set()
    count = 0
    for item in directory.iterdir():
        count += 1
        require(count <= 65536, "LIBVIRT_DESCRIPTOR_LIMIT")
        try:
            target = os.readlink(item)
        except FileNotFoundError:
            continue
        match = re.fullmatch(r"socket:\[([1-9][0-9]*)\]", target)
        if match:
            identities.add(int(match[1]))
    return identities


def diag_packet(raw: bytes, sequence: int) -> tuple[list[DiagRow], bool]:
    rows = []
    complete = False
    index = 0
    while index < len(raw):
        require(index + NL_HEADER.size <= len(raw), "LIBVIRT_SOCKDIAG_HEADER")
        length, kind, flags, current, _ = NL_HEADER.unpack_from(raw, index)
        require(
            NL_HEADER.size <= length <= len(raw) - index
            and current == sequence
            and not (flags & 0x10)
            and not complete,
            "LIBVIRT_SOCKDIAG_INTERRUPTED",
        )
        body = raw[index + NL_HEADER.size : index + length]
        index += (length + 3) & ~3
        if kind == 3:
            require(body in {b"", b"\0" * 4}, "LIBVIRT_SOCKDIAG_STATUS")
            complete = True
            continue
        require(
            kind == SOCK_DIAG_BY_FAMILY and len(body) >= UNIX_MESSAGE.size,
            "LIBVIRT_SOCKDIAG_MESSAGE",
        )
        family, sock_type, state, _, inode, _, _ = UNIX_MESSAGE.unpack_from(body)
        require(family == AF_UNIX, "LIBVIRT_SOCKDIAG_FAMILY")
        if sock_type != socket.SOCK_STREAM:
            continue
        attributes: dict[int, bytes] = {}
        start = UNIX_MESSAGE.size
        while start < len(body):
            require(start + 4 <= len(body), "LIBVIRT_SOCKDIAG_ATTRIBUTE")
            width, key = struct.unpack_from("=HH", body, start)
            require(
                4 <= width <= len(body) - start and key not in attributes,
                "LIBVIRT_SOCKDIAG_ATTRIBUTE",
            )
            attributes[key] = body[start + 4 : start + width]
            start += (width + 3) & ~3
        name = os.fsdecode(attributes.get(UNIX_DIAG_NAME, b"").rstrip(b"\0"))
        peer_raw = attributes.get(UNIX_DIAG_PEER, b"\0" * 4)
        require(len(peer_raw) == 4, "LIBVIRT_SOCKDIAG_PEER")
        vfs = attributes.get(UNIX_DIAG_VFS)
        node = None
        if vfs is not None:
            require(len(vfs) == 8, "LIBVIRT_SOCKDIAG_VFS")
            node = struct.unpack("=II", vfs)
        rows.append((state, inode, name, struct.unpack("=I", peer_raw)[0], node))
    return rows, complete


def sock_diag() -> list[DiagRow]:
    family = getattr(socket, "AF_NETLINK", None)
    require(type(family) is int or isinstance(family, int), "LIBVIRT_SOCKDIAG_UNAVAILABLE")
    assert isinstance(family, int)
    sequence = 1
    request = struct.pack("=BBHIIIII", AF_UNIX, 0, 0, 0xFFFFFFFF, 0, 7, 0xFFFFFFFF, 0xFFFFFFFF)
    header = NL_HEADER.pack(NL_HEADER.size + len(request), SOCK_DIAG_BY_FAMILY, 0x301, sequence, 0)
    all_rows: list[DiagRow] = []
    deadline = time.monotonic() + TIMEOUT
    with socket.socket(family, socket.SOCK_RAW, NETLINK_SOCK_DIAG) as monitor:
        monitor.bind((0, 0))
        monitor.sendto(header + request, (0, 0))
        for _ in range(MAX_DIAG_DATAGRAMS):
            remaining = deadline - time.monotonic()
            require(remaining > 0, "LIBVIRT_SOCKDIAG_TIMEOUT")
            monitor.settimeout(remaining)
            receiver = getattr(monitor, "recvmsg", None)
            require(callable(receiver), "LIBVIRT_SOCKDIAG_UNAVAILABLE")
            assert receiver is not None
            raw, _, flags, address = receiver(256 * 1024)
            truncated = getattr(socket, "MSG_TRUNC", 0x20)
            require(address == (0, 0) and not (flags & truncated), "LIBVIRT_SOCKDIAG_SOURCE")
            rows, complete = diag_packet(raw, sequence)
            all_rows.extend(rows)
            if complete:
                return all_rows
    raise Refusal("LIBVIRT_SOCKDIAG_LIMIT")


def sock_diag_in_namespace(namespace: Path, expected_inode: int) -> list[DiagRow]:
    """Open the monitor in the held client netns without moving the supervisor."""
    setns = getattr(os, "setns", None)
    newnet = getattr(os, "CLONE_NEWNET", None)
    require(callable(setns) and type(newnet) is int, "LIBVIRT_SOCKDIAG_NETNS_UNAVAILABLE")
    assert callable(setns) and type(newnet) is int

    with namespace.open("rb") as descriptor:
        fd = descriptor.fileno()
        held = os.fstat(fd)
        require(held.st_ino == expected_inode, "LIBVIRT_SOCKDIAG_NETNS_CHANGED")

        def inspect() -> list[DiagRow]:
            # setns only moves the calling thread. Linux creates both connected
            # AF_UNIX stream endpoints in the client's netns, even when the
            # filesystem listener lives in the host netns.
            setns(fd, newnet)
            current = (PROC / "thread-self" / "ns" / "net").stat()
            require(
                (current.st_dev, current.st_ino) == (held.st_dev, held.st_ino),
                "LIBVIRT_SOCKDIAG_NETNS_CHANGED",
            )
            return sock_diag()

        with ThreadPoolExecutor(max_workers=1) as worker:
            return worker.submit(inspect).result()


def observe_live_connections(
    host_pid: int, subject_start: int, namespaces: dict[str, int], host: dict[str, Any]
) -> list[str]:
    """Bind an actual connected stream and its accepted peer to the measured daemon."""
    socket_path = host["socket"]
    identity = host["host_socket_identity"]
    daemon = host["daemon"]
    daemon_pid, listener = daemon["pid"], daemon["listener_inode"]
    require(
        socket_path in SOCKETS
        and type(host_pid) is int
        and host_pid > 1
        and host_pid != daemon_pid
        and set(namespaces) == {"mnt", "pid", "net"},
        "LIBVIRT_CONNECTION_SCOPE",
    )
    subject = PROC / str(host_pid) / "ns"
    require(
        all(
            value == (subject / name).stat().st_ino
            and value != (PROC / "self" / "ns" / name).stat().st_ino
            for name, value in namespaces.items()
        )
        and proc_start(host_pid) == subject_start
        and proc_start(daemon_pid) == daemon["start_ticks"]
        and socket_identity(Path(socket_path)) == identity,
        "LIBVIRT_CONNECTION_IDENTITY",
    )
    client_fds = descriptor_sockets(host_pid)
    daemon_fds = descriptor_sockets(daemon_pid)
    rows = sock_diag_in_namespace(subject / "net", namespaces["net"])
    clients = [row for row in rows if row[0] == TCP_ESTABLISHED and row[1] in client_fds]
    servers = [row for row in rows if row[0] == TCP_ESTABLISHED and row[3] in client_fds]
    require(len(clients) == len(servers) == 1, "LIBVIRT_CONNECTION_PATHS_REJECTED")
    _, client_inode, _, client_peer, _ = clients[0]
    _, server_inode, server_path, server_peer, server_vfs = servers[0]
    current_daemon_fds = descriptor_sockets(daemon_pid)
    require(
        server_path == socket_path
        and server_vfs == kernel_socket_node(identity)
        and server_peer == client_inode
        and client_peer == server_inode
        and len({client_inode, server_inode, listener}) == 3
        and {server_inode, listener} <= daemon_fds & current_daemon_fds
        and client_fds == descriptor_sockets(host_pid)
        and proc_start(host_pid) == subject_start
        and proc_start(daemon_pid) == daemon["start_ticks"]
        and all((subject / name).stat().st_ino == value for name, value in namespaces.items())
        and socket_identity(Path(socket_path)) == identity,
        "LIBVIRT_CONNECTION_PATHS_REJECTED",
    )
    return [server_path]


def endpoint_path(path: str) -> bool:
    candidate = Path(path)
    return candidate.is_absolute() and ".." not in candidate.parts and str(candidate) == path


def kernel_socket_node(identity: list[int]) -> tuple[int, int]:
    """UNIX_DIAG_VFS uses a u32 inode and the kernel's 12:20 device encoding."""
    major, minor = getattr(os, "major", None), getattr(os, "minor", None)
    require(callable(major) and callable(minor) and identity[1] < 2**32, "LIBVIRT_SOCKET_VFS")
    assert major is not None and minor is not None
    return identity[1], major(identity[0]) << 20 | minor(identity[0])


def unit_information(daemon: str) -> tuple[int, list[Path]]:
    service, listener = f"{daemon}.service", f"{daemon}-ro.socket"
    result = subprocess.run(
        [
            "/usr/bin/systemctl",
            "--system",
            "--no-pager",
            "show",
            "--property=Id,MainPID,ActiveState,FragmentPath,DropInPaths",
            "--",
            service,
            listener,
        ],
        env=COMMAND_ENV,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=TIMEOUT,
        check=False,
    )
    require(result.returncode == 0 and len(result.stdout) <= CONTROL_LIMIT, "LIBVIRT_SYSTEMD_QUERY")
    units: dict[str, dict[str, str]] = {}
    for record in result.stdout.decode("utf-8").strip().split("\n\n"):
        fields = dict(line.split("=", 1) for line in record.splitlines())
        identity = fields.get("Id", "")
        require(
            identity in {service, listener}
            and identity not in units
            and fields.get("ActiveState") == "active"
            and bool(fields.get("FragmentPath")),
            "LIBVIRT_SYSTEMD_UNIT_STATE",
        )
        units[identity] = fields
    require(set(units) == {service, listener}, "LIBVIRT_SYSTEMD_UNIT_SET")
    pid = int(units[service].get("MainPID", "0"))
    paths: list[Path] = []
    for unit, fields in units.items():
        values = [fields["FragmentPath"], *fields.get("DropInPaths", "").split()]
        require(
            all(endpoint_path(name) for name in values) and Path(values[0]).name == unit,
            "LIBVIRT_SYSTEMD_UNIT_PATHS",
        )
        paths.extend(Path(name) for name in values)
    require(pid > 1 and len(set(paths)) == len(paths), "LIBVIRT_SYSTEMD_MAIN_PROCESS")
    return pid, paths


def daemon_identity(pid: int, daemon: str, started: int) -> tuple[str, str, str]:
    base = PROC / str(pid)
    executable = f"/usr/sbin/{daemon}"
    require(
        os.readlink(base / "exe") == executable and proc_start(pid) == started,
        "LIBVIRT_DAEMON_EXECUTABLE",
    )
    statuses = read_bounded(base / "status").splitlines()
    require(
        any(line.split() == [b"Uid:", b"0", b"0", b"0", b"0"] for line in statuses),
        "LIBVIRT_DAEMON_UID",
    )
    command = read_bounded(base / "cmdline")
    argv = command.split(b"\0")[:-1]
    allowed = {b"--listen", b"--timeout", b"-t", b"--config", b"-f"}
    require(
        command.endswith(b"\0") and bool(argv) and os.fsdecode(argv[0]) == executable,
        "LIBVIRT_DAEMON_CMDLINE",
    )
    it = iter(argv[1:])
    for argument in it:
        require(argument in allowed, "LIBVIRT_DAEMON_CMDLINE_SCOPE")
        if argument in {b"--timeout", b"-t"}:
            require(next(it, b"").isdigit(), "LIBVIRT_DAEMON_CMDLINE_SCOPE")
        if argument in {b"--config", b"-f"}:
            require(
                next(it, b"") == os.fsencode(CONFIG / f"{daemon}.conf"),
                "LIBVIRT_DAEMON_CONFIG_PATH",
            )
    exe = base / "exe"
    with exe.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode)
            and before.st_uid == 0
            and before.st_nlink == 1
            and before.st_mode & 0o022 == 0
            and 0 < before.st_size <= FILE_LIMIT,
            "LIBVIRT_DAEMON_BINARY_METADATA",
        )
        bytes_ = stream.read(before.st_size + 1)
        require(
            stat_identity(before)
            == stat_identity(os.fstat(stream.fileno()))
            == stat_identity(exe.stat())
            and len(bytes_) == before.st_size,
            "LIBVIRT_DAEMON_BINARY_CHANGED",
        )
    require(
        proc_start(pid) == started and read_bounded(base / "cmdline") == command,
        "LIBVIRT_DAEMON_PROCESS_CHANGED",
    )
    return executable, digest(bytes_), digest(command)


def host_configuration(daemon: str, units: list[Path]) -> tuple[dict[str, str], dict[str, str]]:
    paths = [CONFIG / f"{daemon}.conf", CONFIG / "qemu.conf"]
    files = {str(path): digest(protected_file(path)) for path in paths}
    for name in ("network.conf", "virtlogd.conf", "virtlockd.conf"):
        path = CONFIG / name
        if path.exists() or path.is_symlink():
            files[str(path)] = digest(protected_file(path))
    unit_files = {str(path): digest(protected_file(path)) for path in units}
    return files, unit_files


def measure_host(socket_path: str, readonly_implementation_sha256: str) -> dict[str, Any]:
    require(
        socket_path in SOCKETS
        and re.fullmatch(r"[0-9a-f]{64}", readonly_implementation_sha256) is not None,
        "LIBVIRT_HOST_SCOPE",
    )
    daemon = SOCKETS[socket_path]
    pid, unit_paths = unit_information(daemon)
    started = proc_start(pid)
    executable, exe_hash, cmdline = daemon_identity(pid, daemon, started)
    path = Path(socket_path)
    identity = socket_identity(path)
    node = kernel_socket_node(identity)
    descriptor_ids = descriptor_sockets(pid)
    listener = [
        inode
        for state, inode, name, _, vfs in sock_diag()
        if state == TCP_LISTEN and name == socket_path and inode in descriptor_ids and node == vfs
    ]
    require(len(listener) == 1, "LIBVIRT_LISTENER_IDENTITY")
    config, unit_files = host_configuration(daemon, unit_paths)
    with socket.socket(AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(TIMEOUT)
        connection.connect(socket_path)
        peer = peer_credentials(connection)
        require(
            peer[0] in {1, pid}
            and identity == socket_identity(path)
            and proc_start(pid) == started
            and listener[0] in descriptor_sockets(pid)
            and unit_information(daemon) == (pid, unit_paths),
            "LIBVIRT_HOST_CHANGED",
        )
    return {
        "socket": socket_path,
        "host_socket_identity": identity,
        "host_peer": peer,
        "daemon": {
            "pid": pid,
            "start_ticks": started,
            "executable": executable,
            "executable_sha256": exe_hash,
            "cmdline_sha256": cmdline,
            "config": config,
            "unit_files": unit_files,
            "socket_unit": f"{daemon}-ro.socket",
            "service_unit": f"{daemon}.service",
            "listener_inode": listener[0],
            "reviewed_readonly_implementation_sha256": readonly_implementation_sha256,
        },
    }


def measure_view(socket_path: str) -> dict[str, Any]:
    require(socket_path in SOCKETS, "LIBVIRT_VIEW_SCOPE")
    path = Path(socket_path)
    before = socket_identity(path)
    config = protected_file(CLIENT)
    require(config == CLIENT_BYTES, "LIBVIRT_CLIENT_CONFIGURATION")
    with socket.socket(AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(TIMEOUT)
        connection.connect(socket_path)
        peer = peer_credentials(connection)
        require(peer[0] == 0 and before == socket_identity(path), "LIBVIRT_VIEW_SOCKET_DRIFT")
    return {"socket_identity": before, "peer": peer, "client_config_sha256": digest(config)}


def receive(connection: socket.socket, length: int, deadline: float) -> bytes:
    raw = bytearray()
    while len(raw) < length:
        remaining = deadline - time.monotonic()
        require(remaining > 0, "LIBVIRT_RPC_TIMEOUT")
        connection.settimeout(remaining)
        chunk = connection.recv(length - len(raw))
        require(bool(chunk), "LIBVIRT_RPC_TRUNCATED")
        raw += chunk
    return bytes(raw)


def rpc_request(procedure: int, serial: int, payload: bytes) -> bytes:
    header = RPC_HEADER.pack(REMOTE_PROGRAM, REMOTE_VERSION, procedure, 0, serial, 0)
    return RPC_LENGTH.pack(RPC_LENGTH.size + len(header) + len(payload)) + header + payload


def rpc_length(raw: bytes, offset: int = 0) -> int:
    require(len(raw) - offset >= RPC_LENGTH.size, "LIBVIRT_RPC_LENGTH")
    (length,) = RPC_LENGTH.unpack_from(raw, offset)
    require(RPC_LENGTH.size + RPC_HEADER.size <= length <= CONTROL_LIMIT, "LIBVIRT_RPC_LENGTH")
    return int(length)


def rpc_reply(raw: bytes, procedure: int, serial: int) -> tuple[int, bytes]:
    require(rpc_length(raw) == len(raw), "LIBVIRT_RPC_LENGTH")
    program, version, method, kind, reply_serial, status = RPC_HEADER.unpack_from(
        raw, RPC_LENGTH.size
    )
    require(
        (program, version, method, kind, reply_serial)
        == (REMOTE_PROGRAM, REMOTE_VERSION, procedure, 1, serial)
        and status in {0, 1},
        "LIBVIRT_RPC_CORRELATION",
    )
    return status, raw[RPC_LENGTH.size + RPC_HEADER.size :]


def rpc(
    connection: socket.socket,
    procedure: int,
    serial: int,
    payload: bytes,
    deadline: float,
    transcript: list[bytes],
) -> tuple[int, bytes]:
    request = rpc_request(procedure, serial, payload)
    remaining = deadline - time.monotonic()
    require(remaining > 0, "LIBVIRT_RPC_TIMEOUT")
    connection.settimeout(remaining)
    connection.sendall(request)
    prefix = receive(connection, RPC_LENGTH.size, deadline)
    reply = prefix + receive(connection, rpc_length(prefix) - RPC_LENGTH.size, deadline)
    result = rpc_reply(reply, procedure, serial)
    transcript.extend((request, reply))
    return result


def xdr_string(raw: bytes) -> bytes:
    return struct.pack("!I", len(raw)) + raw + b"\0" * ((-len(raw)) % 4)


def denied_error(raw: bytes) -> None:
    """Verify the complete bounded virNetMessageError XDR, including its error code."""
    offset = 0
    reason = "LIBVIRT_FIXTURE_DENIAL_UNPROVEN"

    def word() -> int:
        nonlocal offset
        require(offset + RPC_LENGTH.size <= len(raw), reason)
        (value,) = RPC_LENGTH.unpack_from(raw, offset)
        offset += RPC_LENGTH.size
        return int(value)

    def optional_string() -> None:
        nonlocal offset
        present = word()
        require(present in {0, 1}, reason)
        if present:
            length = word()
            end = offset + ((length + 3) & ~3)
            require(end <= len(raw) and not any(raw[offset + length : end]), reason)
            offset = end

    require(word() == OPERATION_DENIED, reason)
    word()  # error domain
    optional_string()  # message
    word()  # error level
    require(word() == 0, reason)  # unused domain object
    for _ in range(3):
        optional_string()
    word()
    word()
    require(word() == 0 and offset == len(raw), reason)  # unused network object


def check_denial(
    exchange: Callable[[int, int, bytes], tuple[int, bytes]], domain: uuid.UUID
) -> None:
    status, authorization = exchange(AUTH_LIST, 1, b"")
    require(
        status == 0 and authorization in {struct.pack("!I", 0), struct.pack("!II", 1, 0)},
        "LIBVIRT_FIXTURE_AUTH_REJECTED",
    )
    open_args = struct.pack("!I", 1) + xdr_string(b"qemu:///system") + struct.pack("!I", 0)
    status, result = exchange(CONNECT_OPEN, 2, open_args)
    require(status == 0 and not result, "LIBVIRT_FIXTURE_OPEN_REJECTED")
    status, handle = exchange(LOOKUP_BY_UUID, 3, domain.bytes)
    require(status == 0 and len(handle) >= 24, "LIBVIRT_FIXTURE_DOMAIN_REJECTED")
    (length,) = struct.unpack_from("!I", handle)
    end = 4 + ((length + 3) & ~3)
    require(
        0 < length <= 256
        and len(handle) == end + 20
        and handle[end : end + 16] == domain.bytes
        and not any(handle[4 + length : end]),
        "LIBVIRT_FIXTURE_DOMAIN_REJECTED",
    )
    status, denied = exchange(GET_XML_DESC, 4, handle + struct.pack("!I", SECURE_XML))
    require(status == 1, "LIBVIRT_FIXTURE_DENIAL_UNPROVEN")
    denied_error(denied)


def wire_denial(connection: socket.socket, domain: uuid.UUID) -> bytes:
    """Collect the exact sent and received frames over one connected socket."""
    deadline = time.monotonic() + TIMEOUT
    frames: list[bytes] = []
    check_denial(
        lambda method, serial, arg: rpc(connection, method, serial, arg, deadline, frames), domain
    )
    return b"".join(frames)


def trace_header(session: dict[str, str], denial: dict[str, Any]) -> bytes:
    require(
        set(session) == SESSION_FIELDS
        and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in session.values()),
        "LIBVIRT_FIXTURE_SESSION",
    )
    metadata = json.dumps(
        {"session": session, "denial": denial},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    require(0 < len(metadata) <= TRACE_METADATA_LIMIT, "LIBVIRT_FIXTURE_TRACE_METADATA")
    return TRACE_MAGIC + RPC_LENGTH.pack(len(metadata)) + metadata


def verify_denial_trace(raw: bytes, session: dict[str, str], denial: dict[str, Any]) -> None:
    """Use the original frames independently of the collector's interpretation."""
    header = trace_header(session, denial)
    require(
        len(header) < len(raw) <= TRACE_LIMIT and raw.startswith(header),
        "LIBVIRT_FIXTURE_TRACE_BINDING",
    )
    offset = len(header)

    def next_frame() -> bytes:
        nonlocal offset
        end = offset + rpc_length(raw, offset)
        require(end <= len(raw), "LIBVIRT_FIXTURE_TRACE_TRUNCATED")
        frame = raw[offset:end]
        offset = end
        return frame

    def exchange(method: int, serial: int, arg: bytes) -> tuple[int, bytes]:
        require(next_frame() == rpc_request(method, serial, arg), "LIBVIRT_FIXTURE_TRACE_REQUEST")
        return rpc_reply(next_frame(), method, serial)

    domain_uuid = denial["fixture"]["domain_uuid"]
    try:
        domain = uuid.UUID(domain_uuid)
    except (AttributeError, ValueError) as exc:
        raise Refusal("LIBVIRT_FIXTURE_DOMAIN_REJECTED") from exc
    require(str(domain) == domain_uuid, "LIBVIRT_FIXTURE_DOMAIN_REJECTED")
    check_denial(exchange, domain)
    require(offset == len(raw), "LIBVIRT_FIXTURE_TRACE_TRAILING")


def collect_fixture_denial(
    path: Path,
    disposable_pid: int,
    domain_uuid: str,
    host: dict[str, Any],
    *,
    output: Path,
    session: dict[str, str],
) -> dict[str, Any]:
    """Save original frames tied to this authorized session and disposable daemon."""
    domain = uuid.UUID(domain_uuid)
    require(
        str(domain) == domain_uuid
        and endpoint_path(str(path))
        and str(path) not in SOCKETS
        and type(disposable_pid) is int
        and disposable_pid > 1
        and host["socket"] in SOCKETS
        and disposable_pid != host["daemon"]["pid"],
        "LIBVIRT_FIXTURE_SCOPE",
    )
    info = path.parent.lstat()
    require(
        stat.S_ISDIR(info.st_mode)
        and info.st_uid == 0
        and stat.S_IMODE(info.st_mode) == 0o700
        and path.parent.resolve(strict=True) == path.parent,
        "LIBVIRT_FIXTURE_SOCKET_DIRECTORY",
    )
    daemon = host["daemon"]
    base = PROC / str(disposable_pid)
    started = proc_start(disposable_pid)
    names = ("mnt", "pid", "net")
    host_namespaces = {name: (PROC / "self" / "ns" / name).stat().st_ino for name in names}
    namespaces = {name: (base / "ns" / name).stat().st_ino for name in names}
    require(
        all(namespaces[name] != host_namespaces[name] for name in names),
        "LIBVIRT_FIXTURE_NAMESPACE_REQUIRED",
    )

    def measured_daemon() -> tuple[str, str, str]:
        observed = daemon_identity(disposable_pid, SOCKETS[host["socket"]], started)
        require(
            observed[0:2] == (daemon["executable"], daemon["executable_sha256"]),
            "LIBVIRT_FIXTURE_BINARY_MISMATCH",
        )
        for paths in (daemon["config"], daemon["unit_files"]):
            for name, expected in paths.items():
                require(
                    digest(protected_file(Path(name), base / "root")) == expected,
                    "LIBVIRT_FIXTURE_CONFIGURATION_MISMATCH",
                )
        return observed

    executable, exe_hash, cmdline = measured_daemon()
    identity = socket_identity(path)
    require(identity[:2] != host["host_socket_identity"][:2], "LIBVIRT_FIXTURE_SOCKET_ALIAS")
    with socket.socket(AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(TIMEOUT)
        connection.connect(str(path))
        peer = peer_credentials(connection)
        connection_stat = os.fstat(connection.fileno())
        require(
            peer[0] == disposable_pid
            and socket_identity(path) == identity
            and stat.S_ISSOCK(connection_stat.st_mode)
            and connection_stat.st_ino > 0,
            "LIBVIRT_FIXTURE_PEER_REJECTED",
        )
        wire = wire_denial(connection, domain)
        require(
            peer_credentials(connection) == peer
            and socket_identity(path) == identity
            and proc_start(disposable_pid) == started
            and all((base / "ns" / name).stat().st_ino == namespaces[name] for name in names)
            and measured_daemon() == (executable, exe_hash, cmdline),
            "LIBVIRT_FIXTURE_CHANGED",
        )
    denial = {
        "scope": "DISPOSABLE_FIXTURE",
        "daemon_sha256": exe_hash,
        "reviewed_readonly_implementation_sha256": daemon[
            "reviewed_readonly_implementation_sha256"
        ],
        "config": daemon["config"],
        "unit_files": daemon["unit_files"],
        "client_connect_flags": 0,
        "fixture": {
            "socket": str(path),
            "socket_identity": identity,
            "connection_identity": [connection_stat.st_dev, connection_stat.st_ino],
            "peer": peer,
            "start_ticks": started,
            "namespaces": namespaces,
            "host_namespaces": host_namespaces,
            "executable": executable,
            "cmdline_sha256": cmdline,
            "domain_uuid": domain_uuid,
        },
    }
    artifact = output / "fixture-readonly.rpc"
    raw = trace_header(session, denial) + wire
    with os.fdopen(os.open(artifact, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(raw)
    return {**denial, "trace": {"path": artifact.name, "sha256": digest(raw)}}


def assemble(
    before: dict[str, Any],
    after: dict[str, Any],
    view: dict[str, Any],
    read_results: dict[str, str],
    connect_paths: list[str],
    denial: dict[str, Any],
) -> dict[str, Any]:
    require(
        before == after
        and set(view) == {"socket_identity", "peer", "client_config_sha256"}
        and view["socket_identity"] == before["host_socket_identity"]
        and view["peer"] == [0, *before["host_peer"][1:]]
        and view["client_config_sha256"] == digest(CLIENT_BYTES)
        and connect_paths == [before["socket"]],
        "LIBVIRT_EVIDENCE_CHANGED",
    )
    return {
        **before,
        **view,
        "uri": "qemu:///system",
        "connect_paths": connect_paths,
        "read_results": read_results,
        "fixture_denial": denial,
    }
