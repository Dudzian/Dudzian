"""Synthetic libvirt RPC and host evidence tests using no libvirt daemon."""

from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import uuid

import pytest

from scripts import stage9_v5d11_libvirt as virt

HOST_SOCKET = "/run/libvirt/libvirt-sock-ro"
SHA = "ab" * 32
DOMAIN = uuid.UUID("c1cf34ad-539d-4669-899a-6859ea57e925")
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux-only observer evidence")


def reply(connection, message, frames, *, error=0, serial_shift=0):
    header = virt.receive(connection, 4, time.monotonic() + 2)
    (length,) = struct.unpack("!I", header)
    raw = virt.receive(connection, length - 4, time.monotonic() + 2)
    program, version, procedure, kind, serial, status = virt.RPC_HEADER.unpack_from(raw)
    assert (program, version, kind, status) == (virt.REMOTE_PROGRAM, 1, 0, 0)
    output = virt.RPC_HEADER.pack(program, version, procedure, 1, serial + serial_shift, error)
    response = struct.pack("!I", len(output) + len(message) + 4) + output + message
    connection.sendall(response)
    frames.extend((header + raw, response))
    return procedure, raw[virt.RPC_HEADER.size :]


@pytest.mark.parametrize(
    ("variant", "reason"),
    [
        ("ok", ""),
        ("peer", "CORRELATION"),
        ("auth", "AUTH_REJECTED"),
        ("domain", "DOMAIN_REJECTED"),
        ("allowed", "DENIAL_UNPROVEN"),
        ("other_error", "DENIAL_UNPROVEN"),
        ("incomplete_error", "DENIAL_UNPROVEN"),
        ("bad_padding", "DENIAL_UNPROVEN"),
    ],
)
def test_actual_socketpair_only_sends_read_operations(variant, reason):
    client, server = socket.socketpair()
    with client, server:
        handle = virt.xdr_string(b"disposable") + DOMAIN.bytes + b"\xff" * 4
        frames = []

        def serve():
            methods = []
            methods.append(
                reply(
                    server,
                    struct.pack("!II", 1, 2 if variant == "auth" else 0),
                    frames,
                    serial_shift=int(variant == "peer"),
                )
            )
            if variant in {"auth", "peer"}:
                return methods
            methods.append(reply(server, b"", frames))
            methods.append(reply(server, handle + (b"x" if variant == "domain" else b""), frames))
            if variant == "domain":
                return methods
            code = 1 if variant == "other_error" else virt.OPERATION_DENIED
            message = b"deny!" if variant != "bad_padding" else b"deny!\0\0x"
            error = struct.pack("!III", code, 0, 1)
            error += (
                virt.xdr_string(message) if variant != "bad_padding" else b"\0\0\0\x05" + message
            )
            error += struct.pack("!8I", 2, 0, 0, 0, 0, 0, 0, 0)
            if variant == "incomplete_error":
                error = error[:8]
            methods.append(reply(server, error, frames, error=int(variant != "allowed")))
            return methods

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as workers:
            future = workers.submit(serve)
            if reason:
                with pytest.raises(virt.Refusal, match=reason):
                    virt.wire_denial(client, DOMAIN)
            else:
                transcript = virt.wire_denial(client, DOMAIN)
            assert (
                future.result(timeout=2)
                == [
                    (virt.AUTH_LIST, b""),
                    (
                        virt.CONNECT_OPEN,
                        b"\0\0\0\x01" + virt.xdr_string(b"qemu:///system") + b"\0" * 4,
                    ),
                    (virt.LOOKUP_BY_UUID, DOMAIN.bytes),
                    (virt.GET_XML_DESC, handle + b"\0\0\0\x01"),
                ][: {"auth": 1, "peer": 1, "domain": 3}.get(variant, 4)]
            )
            if not reason:
                assert transcript == b"".join(frames)


def test_receive_fails_for_truncated_and_expired_socket():
    client, server = socket.socketpair()
    with client, server:
        server.shutdown(socket.SHUT_WR)
        with pytest.raises(virt.Refusal, match="RPC_TRUNCATED"):
            virt.receive(client, 4, time.monotonic() + 2)
        with pytest.raises(virt.Refusal, match="RPC_TIMEOUT"):
            virt.receive(client, 4, 0)


def netlink(kind, data, flags=2, sequence=1):
    return virt.NL_HEADER.pack(virt.NL_HEADER.size + len(data), kind, flags, sequence, 0) + data


def attr(key, data):
    count = len(data) + 4
    return struct.pack("=HH", count, key) + data + b"\0" * ((-count) % 4)


def sockdiag_packet():
    base = virt.UNIX_MESSAGE.pack(virt.AF_UNIX, socket.SOCK_STREAM, 1, 0, 17, 0, 0)
    base += attr(0, HOST_SOCKET.encode() + b"\0") + attr(2, struct.pack("=I", 15))
    return netlink(virt.SOCK_DIAG_BY_FAMILY, base) + netlink(3, b"\0" * 4)


@pytest.mark.parametrize(
    "change", ["none", "truncated", "interrupt", "unexpected", "bad_attribute", "bad_end"]
)
def test_sockdiag_decoder(change):
    good = sockdiag_packet()
    if change == "none":
        assert virt.diag_packet(good, 1) == ([(1, 17, HOST_SOCKET, 15, None)], True)
    else:
        raw = {
            "truncated": good[:-2],
            "interrupt": netlink(3, b"", flags=0x12),
            "unexpected": netlink(4, b""),
            "bad_end": netlink(3, b"\0" * 8),
            "bad_attribute": netlink(20, virt.UNIX_MESSAGE.pack(1, 1, 1, 0, 17, 0, 0) + b"\0" * 4),
        }[change]
        with pytest.raises(virt.Refusal, match="SOCKDIAG"):
            virt.diag_packet(raw, 1)


class SocketDouble:
    def __init__(self, response):
        self.response = response
        self.path = None
        self.timeout = None
        self.bound = None
        self.sent = b""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def settimeout(self, timeout):
        self.timeout = timeout

    def connect(self, path):
        self.path = path

    def bind(self, address):
        self.bound = address

    def sendto(self, data, address):
        assert address == (0, 0)
        self.sent = data

    def recvmsg(self, capacity):
        assert len(self.response) <= capacity
        return self.response, [], 0, (0, 0)

    def getsockopt(self, level, option, count):
        assert level == socket.SOL_SOCKET and option == socket.SO_PEERCRED and count == 12
        return self.response


def test_sockdiag_dump(monkeypatch):
    monitor = SocketDouble(sockdiag_packet())
    monkeypatch.setattr(socket, "socket", lambda *args: monitor)
    assert virt.sock_diag() == [(1, 17, HOST_SOCKET, 15, None)]
    length, kind, flags, sequence, _ = virt.NL_HEADER.unpack_from(monitor.sent)
    assert (length, kind, flags, sequence) == (len(monitor.sent), 20, 0x301, 1)
    assert monitor.bound == (0, 0)


@pytest.mark.parametrize("case", ["ok", "wrong_handle", "wrong_thread_netns"])
def test_sockdiag_uses_held_namespace_in_dedicated_thread(monkeypatch, tmp_path, case):
    namespace = tmp_path / "net"
    namespace.touch()
    expected = namespace.stat()
    worker_ns = tmp_path / "thread-self" / "ns"
    worker_ns.mkdir(parents=True)
    (worker_ns / "net").touch()
    caller = threading.get_ident()
    called = []

    def enter(fd, kind):
        assert kind == 0x40000000 and threading.get_ident() != caller
        held = os.fstat(fd)
        assert (held.st_dev, held.st_ino) == (expected.st_dev, expected.st_ino)
        called.append("setns")
        if case != "wrong_thread_netns":
            (worker_ns / "net").unlink()
            (worker_ns / "net").symlink_to(namespace)

    def dump():
        assert threading.get_ident() != caller and called == ["setns"]
        called.append("dump")
        return [(1, 17, HOST_SOCKET, 15, None)]

    monkeypatch.setattr(os, "setns", enter, raising=False)
    monkeypatch.setattr(os, "CLONE_NEWNET", 0x40000000, raising=False)
    monkeypatch.setattr(virt, "PROC", tmp_path)
    monkeypatch.setattr(virt, "sock_diag", dump)
    if case == "ok":
        assert virt.sock_diag_in_namespace(namespace, expected.st_ino) == [
            (1, 17, HOST_SOCKET, 15, None)
        ]
        assert called == ["setns", "dump"]
    else:
        with pytest.raises(virt.Refusal, match="SOCKDIAG_NETNS_CHANGED"):
            virt.sock_diag_in_namespace(namespace, expected.st_ino + (case == "wrong_handle"))
        assert called == ([] if case == "wrong_handle" else ["setns"])


@pytest.mark.parametrize(
    "change",
    [
        "ok",
        "bad_subject",
        "bad_path",
        "bad_client_start",
        "bad_daemon_start",
        "bad_netns",
        "host_netns",
        "socket_drift",
        "no_listener",
        "wrong_daemon",
        "changed_client_fds",
        "changed_process",
        "changed_namespace",
        "no_client",
        "different_socket",
        "different_vfs",
        "nonreciprocal",
        "ambiguous_client",
        "ambiguous_server",
    ],
)
def test_connection_requires_reciprocal_kernel_peers_and_exact_identities(
    monkeypatch, tmp_path, change
):
    subject_ns = tmp_path / "3" / "ns"
    host_ns = tmp_path / "self" / "ns"
    namespaces = {}
    for directory in (subject_ns, host_ns):
        directory.mkdir(parents=True)
        for name in ("mnt", "pid", "net"):
            (directory / name).touch()
    for item in subject_ns.iterdir():
        namespaces[item.name] = item.stat().st_ino
    if change == "bad_netns":
        namespaces["net"] += 1
    elif change == "host_netns":
        (subject_ns / "net").unlink()
        (subject_ns / "net").symlink_to(host_ns / "net")
        namespaces["net"] = (host_ns / "net").stat().st_ino

    identity = [os.makedev(8, 2), 90, 0, 0, 0o666]
    node = virt.kernel_socket_node(identity)
    daemon = {"pid": 42, "start_ticks": 304, "listener_inode": 23}
    host = {
        "socket": HOST_SOCKET if change != "bad_path" else "/run/libvirt/libvirt-sock",
        "host_socket_identity": identity,
        "daemon": daemon,
    }
    started = 201
    rows = [(1, 15, "", 17, None), (1, 17, HOST_SOCKET, 15, node)]
    if change == "different_socket":
        rows[1] = (1, 17, "/run/libvirt/virtqemud-sock-ro", 15, node)
    elif change == "different_vfs":
        rows[1] = (1, 17, HOST_SOCKET, 15, (91, node[1]))
    elif change == "nonreciprocal":
        rows[0] = (1, 15, "", 18, None)
    elif change == "no_client":
        del rows[0]
    elif change == "ambiguous_client":
        rows.append((1, 16, "", 18, None))
    elif change == "ambiguous_server":
        rows.append((1, 18, HOST_SOCKET, 15, node))

    queried = False

    def dump(namespace, inode):
        nonlocal queried
        assert (namespace, inode) == (subject_ns / "net", namespaces["net"])
        queried = True
        if change == "changed_namespace":
            replacement = subject_ns / "replacement"
            replacement.touch()
            replacement.rename(subject_ns / "net")
        return rows

    def pid_start(pid):
        if pid == 3:
            return started + (
                change == "bad_client_start" or queried and change == "changed_process"
            )
        assert pid == 42
        return 304 + (change == "bad_daemon_start")

    def descriptors(pid):
        if pid == 3:
            return (
                {15, 16}
                if change == "ambiguous_client" or (queried and change == "changed_client_fds")
                else {15}
            )
        assert pid == 42
        return {"no_listener": {17}, "wrong_daemon": {23}}.get(change, {17, 23})

    monkeypatch.setattr(virt, "PROC", tmp_path)
    monkeypatch.setattr(virt, "proc_start", pid_start)
    monkeypatch.setattr(virt, "descriptor_sockets", descriptors)
    monkeypatch.setattr(virt, "sock_diag_in_namespace", dump)
    monkeypatch.setattr(
        virt, "socket_identity", lambda path: identity if change != "socket_drift" else []
    )
    if change == "ok":
        assert virt.observe_live_connections(3, started, namespaces, host) == [HOST_SOCKET]
        return
    reasons = {
        "bad_subject": "CONNECTION_SCOPE",
        "bad_path": "CONNECTION_SCOPE",
        "bad_client_start": "CONNECTION_IDENTITY",
        "bad_daemon_start": "CONNECTION_IDENTITY",
        "bad_netns": "CONNECTION_IDENTITY",
        "host_netns": "CONNECTION_IDENTITY",
        "socket_drift": "CONNECTION_IDENTITY",
    }
    with pytest.raises(virt.Refusal, match=reasons.get(change, "CONNECTION_PATHS_REJECTED")):
        virt.observe_live_connections(
            42 if change == "bad_subject" else 3, started, namespaces, host
        )


def test_proc_start_and_fds_without_mocked_kernel(tmp_path):
    target = tmp_path / "42"
    target.mkdir()
    (target / "stat").write_bytes(
        b"42 (Name with )) S " + b" ".join([b"1"] * 18 + [b"921"]) + b"\n"
    )
    fds = target / "fd"
    fds.mkdir()
    (fds / "10").symlink_to("socket:[99]")
    (fds / "3").symlink_to("/dev/null")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(virt, "PROC", tmp_path)
        assert virt.proc_start(42) == 921
        assert virt.descriptor_sockets(42) == {99}
        (target / "stat").write_bytes(b"partial")
        with pytest.raises(virt.Refusal, match="PROCESS_STAT"):
            virt.proc_start(42)


def test_unit_information_requires_active_exact_units(monkeypatch):
    def run(argv, **kwargs):
        assert argv[1:4] == ["--system", "--no-pager", "show"]
        assert kwargs["env"] == virt.COMMAND_ENV and kwargs["timeout"] == virt.TIMEOUT
        return SimpleNamespace(returncode=0, stdout=text)

    text = (
        b"Id=libvirtd.service\nActiveState=active\nMainPID=42\n"
        b"FragmentPath=/usr/lib/systemd/system/libvirtd.service\nDropInPaths=/etc/systemd/system/libvirtd.service.d/one.conf\n\n"
        b"Id=libvirtd-ro.socket\nActiveState=active\nFragmentPath=/usr/lib/systemd/system/libvirtd-ro.socket\nDropInPaths=\n"
    )
    monkeypatch.setattr(subprocess, "run", run)
    pid, files = virt.unit_information("libvirtd")
    assert pid == 42 and [path.name for path in files] == [
        "libvirtd.service",
        "one.conf",
        "libvirtd-ro.socket",
    ]
    text = text.replace(b"ActiveState=active", b"ActiveState=inactive")
    with pytest.raises(virt.Refusal, match="UNIT_STATE"):
        virt.unit_information("libvirtd")


def test_protected_file_opens_exact_path_and_refuses_symlink(tmp_path, monkeypatch):
    directory = tmp_path / "config"
    directory.mkdir(mode=0o700)
    target = directory / "live.conf"
    target.write_bytes(b"a=b")
    target.chmod(0o600)
    actual = os.fstat, os.stat

    def root_owned(info):
        return SimpleNamespace(
            **{
                name: (0 if name == "st_uid" else getattr(info, name))
                for name in dir(info)
                if name.startswith("st_")
            }
        )

    monkeypatch.setattr(os, "fstat", lambda fd: root_owned(actual[0](fd)))
    monkeypatch.setattr(os, "stat", lambda *args, **kwargs: root_owned(actual[1](*args, **kwargs)))
    assert virt.protected_file(Path("/config/live.conf"), tmp_path) == b"a=b"
    target.chmod(0o666)
    with pytest.raises(virt.Refusal, match="FILE_METADATA"):
        virt.protected_file(Path("/config/live.conf"), tmp_path)
    target.chmod(0o600)
    (directory / "alias").symlink_to("live.conf")
    with pytest.raises(OSError):
        virt.protected_file(Path("/config/alias"), tmp_path)


def host_ready(monkeypatch, tmp_path):
    daemon_path = f"/usr/sbin/{virt.SOCKETS[HOST_SOCKET]}"
    socket = SocketDouble(struct.pack("=iII", 42, 0, 9))
    monkeypatch.setattr(virt.socket, "socket", lambda *_: socket)
    monkeypatch.setattr(virt, "unit_information", lambda _: (42, [Path("/libvirtd.service")]))
    monkeypatch.setattr(virt, "proc_start", lambda _: 100)
    monkeypatch.setattr(virt, "daemon_identity", lambda *args: (daemon_path, SHA, SHA))
    monkeypatch.setattr(
        virt,
        "socket_identity",
        lambda path: [5, 90 if str(path) == HOST_SOCKET else 91, 0, 9, 0o660],
    )
    monkeypatch.setattr(virt, "descriptor_sockets", lambda pid: {23})
    monkeypatch.setattr(virt, "sock_diag", lambda: [(10, 23, HOST_SOCKET, 0, (90, 5))])
    monkeypatch.setattr(
        virt,
        "host_configuration",
        lambda *a: ({"/etc/libvirt/libvirtd.conf": SHA}, {"/libvirtd.service": SHA}),
    )
    return socket, virt.measure_host(HOST_SOCKET, SHA)


def test_host_view_fixture_assembly(tmp_path, monkeypatch):
    connection, host = host_ready(monkeypatch, tmp_path)
    assert connection.path == HOST_SOCKET
    assert host["host_peer"] == [42, 0, 9] and host["daemon"]["listener_inode"] == 23
    after = virt.measure_host(HOST_SOCKET, SHA)
    connection.response = struct.pack("=iII", 0, 0, 9)
    monkeypatch.setattr(virt, "protected_file", lambda *a: virt.CLIENT_BYTES)
    view = virt.measure_view(HOST_SOCKET)
    assert (
        virt.assemble(host, after, view, {"close": "PASS"}, [HOST_SOCKET], {})["uri"]
        == "qemu:///system"
    )
    with pytest.raises(virt.Refusal, match="EVIDENCE_CHANGED"):
        virt.assemble(host, after, view, {}, ["/wrong"], {})
    monkeypatch.setattr(virt, "descriptor_sockets", lambda _: set())
    with pytest.raises(virt.Refusal, match="LISTENER_IDENTITY"):
        virt.measure_host(HOST_SOCKET, SHA)


def test_daemon_identity_hashes_open_exe_and_command(tmp_path, monkeypatch):
    process = tmp_path / "42"
    process.mkdir()
    (process / "status").write_bytes(b"Name: python\nUid: 0 0 0 0\n")
    command = b"/usr/sbin/libvirtd\0--timeout\0" + b"120\0"
    (process / "cmdline").write_bytes(command)
    binary = process / "exe"
    binary.write_bytes(b"synthetic trusted bytes")
    binary.chmod(0o600)
    monkeypatch.setattr(virt, "PROC", tmp_path)
    monkeypatch.setattr(virt, "proc_start", lambda pid: 7)
    monkeypatch.setattr(os, "readlink", lambda path: "/usr/sbin/libvirtd")
    stat_before, fstat_before = Path.stat, os.fstat

    def owner(info):
        return SimpleNamespace(
            **{
                name: (0 if name == "st_uid" else getattr(info, name))
                for name in dir(info)
                if name.startswith("st_")
            }
        )

    monkeypatch.setattr(Path, "stat", lambda p, **kw: owner(stat_before(p, **kw)))
    monkeypatch.setattr(os, "fstat", lambda fd: owner(fstat_before(fd)))
    assert virt.daemon_identity(42, "libvirtd", 7) == (
        "/usr/sbin/libvirtd",
        virt.digest(binary.read_bytes()),
        virt.digest(command),
    )
    (process / "cmdline").write_bytes(b"/usr/sbin/libvirtd\0--unknown\0")
    with pytest.raises(virt.Refusal, match="CMDLINE_SCOPE"):
        virt.daemon_identity(42, "libvirtd", 7)


def test_fixture_namespace_binary_config_and_peer_guards(tmp_path, monkeypatch):
    connection, host = host_ready(monkeypatch, tmp_path)
    path = tmp_path / "private" / "fixture-sock"
    path.parent.mkdir(mode=0o700)
    actual = Path.lstat
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda p: SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_uid=0)
        if p == path.parent
        else actual(p),
    )
    monkeypatch.setattr(Path, "resolve", lambda p, **kw: p)
    stat_actual = Path.stat
    namespace = 92
    monkeypatch.setattr(
        Path,
        "stat",
        lambda p, **kw: SimpleNamespace(st_ino=(91 if "/self/" in str(p) else namespace))
        if "/ns/" in str(p)
        else stat_actual(p, **kw),
    )
    actual_fstat = os.fstat
    monkeypatch.setattr(connection, "fileno", lambda: 999, raising=False)
    monkeypatch.setattr(
        os,
        "fstat",
        lambda fd: SimpleNamespace(st_mode=stat.S_IFSOCK, st_dev=9, st_ino=404)
        if fd == 999
        else actual_fstat(fd),
    )
    for role in ("config", "unit_files"):
        host["daemon"][role] = dict.fromkeys(host["daemon"][role], virt.digest(b"same"))
    monkeypatch.setattr(virt, "protected_file", lambda *a: b"same")
    wire = []
    session = dict.fromkeys(virt.SESSION_FIELDS, SHA)

    def collect(conn, domain):
        wire.append((conn, domain))
        return b"synthetic-source-frames"

    monkeypatch.setattr(virt, "wire_denial", collect)
    connection.response = struct.pack("=iII", 43, 0, 9)
    evidence = virt.collect_fixture_denial(
        path, 43, str(DOMAIN), host, output=tmp_path, session=session
    )
    assert evidence["scope"] == "DISPOSABLE_FIXTURE" and wire == [(connection, DOMAIN)]
    assert evidence["fixture"]["connection_identity"] == [9, 404]
    trace = tmp_path / evidence["trace"]["path"]
    assert stat.S_IMODE(trace.stat().st_mode) == 0o600
    raw = trace.read_bytes()
    assert raw.endswith(b"synthetic-source-frames") and evidence["trace"]["sha256"] == virt.digest(
        raw
    )
    with pytest.raises(FileExistsError):
        virt.collect_fixture_denial(path, 43, str(DOMAIN), host, output=tmp_path, session=session)
    connection.response = struct.pack("=iII", 44, 0, 9)
    with pytest.raises(virt.Refusal, match="FIXTURE_PEER_REJECTED"):
        virt.collect_fixture_denial(path, 43, str(DOMAIN), host, output=tmp_path, session=session)
    namespace = 91
    with pytest.raises(virt.Refusal, match="NAMESPACE_REQUIRED"):
        virt.collect_fixture_denial(path, 43, str(DOMAIN), host, output=tmp_path, session=session)
    with pytest.raises(virt.Refusal, match="FIXTURE_SCOPE"):
        virt.collect_fixture_denial(path, 42, str(DOMAIN), host, output=tmp_path, session=session)
