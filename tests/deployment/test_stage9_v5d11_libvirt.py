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
import time
from types import SimpleNamespace
import uuid

import pytest

from scripts import stage9_v5d11_libvirt as virt

HOST_SOCKET = "/run/libvirt/libvirt-sock-ro"
SHA = "ab" * 32
DOMAIN = uuid.UUID("c1cf34ad-539d-4669-899a-6859ea57e925")
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux-only observer evidence")


def reply(connection, message, *, error=0, serial_shift=0):
    header = virt.receive(connection, 4, time.monotonic() + 2)
    (length,) = struct.unpack("!I", header)
    raw = virt.receive(connection, length - 4, time.monotonic() + 2)
    program, version, procedure, kind, serial, status = virt.RPC_HEADER.unpack_from(raw)
    assert (program, version, kind, status) == (virt.REMOTE_PROGRAM, 1, 0, 0)
    output = virt.RPC_HEADER.pack(program, version, procedure, 1, serial + serial_shift, error)
    connection.sendall(struct.pack("!I", len(output) + len(message) + 4) + output + message)
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
    ],
)
def test_actual_socketpair_only_sends_read_operations(variant, reason):
    client, server = socket.socketpair()
    with client, server:
        handle = virt.xdr_string(b"disposable") + DOMAIN.bytes + b"\xff" * 4

        def serve():
            methods = []
            methods.append(
                reply(
                    server,
                    struct.pack("!II", 1, 2 if variant == "auth" else 0),
                    serial_shift=int(variant == "peer"),
                )
            )
            if variant in {"auth", "peer"}:
                return methods
            methods.append(reply(server, b""))
            methods.append(reply(server, handle + (b"x" if variant == "domain" else b"")))
            if variant == "domain":
                return methods
            code = 1 if variant == "other_error" else virt.OPERATION_DENIED
            methods.append(
                reply(server, struct.pack("!II", code, 0), error=int(variant != "allowed"))
            )
            return methods

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as workers:
            future = workers.submit(serve)
            if reason:
                with pytest.raises(virt.Refusal, match=reason):
                    virt.wire_denial(client, DOMAIN)
            else:
                virt.wire_denial(client, DOMAIN)
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


def test_sockdiag_dump_and_connection_correlation(monkeypatch):
    monitor = SocketDouble(sockdiag_packet())
    monkeypatch.setattr(socket, "socket", lambda *args: monitor)
    monkeypatch.setattr(virt, "proc_start", lambda _: 100)
    monkeypatch.setattr(virt, "descriptor_sockets", lambda _: {15})
    assert virt.observe_live_connections(3, HOST_SOCKET) == [HOST_SOCKET]
    length, kind, flags, sequence, _ = virt.NL_HEADER.unpack_from(monitor.sent)
    assert (length, kind, flags, sequence) == (len(monitor.sent), 20, 0x301, 1)
    assert monitor.bound == (0, 0)
    with pytest.raises(virt.Refusal, match="PATHS_REJECTED"):
        virt.observe_live_connections(3, "/run/libvirt/libvirt-sock")
    with pytest.raises(virt.Refusal, match="CONNECTION_SCOPE"):
        virt.observe_live_connections(1, HOST_SOCKET)


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
    monkeypatch.setattr(virt, "digest", lambda raw: SHA)
    monkeypatch.setattr(virt, "protected_file", lambda *a: b"same")
    wire = []
    monkeypatch.setattr(virt, "wire_denial", lambda conn, domain: wire.append((conn, domain)))
    connection.response = struct.pack("=iII", 43, 0, 9)
    evidence = virt.collect_fixture_denial(path, 43, str(DOMAIN), host)
    assert evidence["scope"] == "DISPOSABLE_FIXTURE" and wire == [(connection, DOMAIN)]
    connection.response = struct.pack("=iII", 44, 0, 9)
    with pytest.raises(virt.Refusal, match="FIXTURE_PEER_REJECTED"):
        virt.collect_fixture_denial(path, 43, str(DOMAIN), host)
    namespace = 91
    with pytest.raises(virt.Refusal, match="NAMESPACE_REQUIRED"):
        virt.collect_fixture_denial(path, 43, str(DOMAIN), host)
    with pytest.raises(virt.Refusal, match="FIXTURE_SCOPE"):
        virt.collect_fixture_denial(path, 42, str(DOMAIN), host)
