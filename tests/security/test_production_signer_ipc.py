"""Protocol/process tests; simulated metadata never proves account isolation."""

from __future__ import annotations

import multiprocessing
import os
import socket
import stat
import struct
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from bot_core import production_signer_ipc as ipc


@pytest.fixture
def policy(tmp_path):
    return ipc.SignerSocketPolicy(
        "root_signer",
        str(tmp_path / "signer.sock"),
        os.getuid(),
        os.getgid(),
        os.getuid() + 10000,
        os.getgid() + 10000,
        "a" * 64,
    )


def request(policy, **changes):
    result = {
        "version": 1,
        "operation": "DIAGNOSTIC",
        "role": policy.role,
        "installation_sha256": policy.installation_sha256,
        "nonce": "b" * 64,
    }
    result.update(changes)
    return result


def response(value):
    return {
        **value,
        "status": "BLOCKED",
        "production_emission": "BLOCKED",
        "facts": {"private_control_status": "BLOCKED"},
        "blockers": ["PRIVATE_CONTROL_NOT_INDEPENDENTLY_PROVEN"],
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"version": True},
        {"version": 2},
        {"operation": "SIGN"},
        {"operation": []},
        {"role": "history_signer"},
        {"installation_sha256": "c" * 64},
        {"nonce": "B" * 64},
        {"nonce": 1},
        {"credential": "caller-controlled"},
    ],
)
def test_exact_diagnostic_request_rejects_unsafe_fields(policy, changes):
    with pytest.raises(ipc.ProductionSignerIPCError, match="EXACT_DIAGNOSTIC"):
        ipc._request(request(policy, **changes), policy)


@pytest.mark.parametrize(
    "changes",
    [
        {"version": True},
        {"status": "PASS"},
        {"production_emission": "PASS"},
        {"facts": []},
        {"blockers": []},
        {"blockers": [1]},
        {"nonce": "c" * 64},
        {"extra": "unversioned"},
    ],
)
def test_response_cannot_claim_readiness_or_change_nonce(policy, changes):
    original = request(policy)
    value = response(original)
    value.update(changes)
    with pytest.raises(ipc.ProductionSignerIPCError, match="EXACT_BLOCKED"):
        ipc._response(value, original)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"version":1,"version":1}',
        b'{"facts":{"a":1,"a":2}}',
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b'{"value":1e999}',
        b"[]",
        b"null",
        b"\xff",
        b"{",
        b'{"a":1} trailing',
    ],
)
def test_frame_strict_json(raw):
    left, right = socket.socketpair()
    with left, right:
        left.sendall(struct.pack("!I", len(raw)) + raw)
        with pytest.raises(ipc.ProductionSignerIPCError):
            ipc._receive_frame(right, time.monotonic() + 1)


@pytest.mark.parametrize("size", [0, ipc.MAX_FRAME_BYTES + 1, 2**32 - 1])
def test_oversized_header_rejected_before_payload(size):
    left, right = socket.socketpair()
    with left, right:
        left.sendall(struct.pack("!I", size))
        with pytest.raises(ipc.ProductionSignerIPCError, match="BOUNDED"):
            ipc._receive_frame(right, time.monotonic() + 1)


def test_disconnect_and_total_deadline_fail_closed():
    left, right = socket.socketpair()
    with left, right:
        left.sendall(struct.pack("!I", 100) + b"{")
        with pytest.raises(TimeoutError):
            ipc._receive_frame(right, time.monotonic() + 0.03)
        left.shutdown(socket.SHUT_WR)
        with pytest.raises(ipc.ProductionSignerIPCError, match="DISCONNECTED"):
            ipc._receive_exact(right, 1, time.monotonic() + 1)
        with pytest.raises(ipc.ProductionSignerIPCError, match="DEADLINE"):
            ipc._receive_exact(right, 1, time.monotonic() - 1)


def test_multiple_frames_rejected():
    left, right = socket.socketpair()
    with left, right:
        ipc._send_frame(left, {"one": 1}, time.monotonic() + 1)
        ipc._send_frame(left, {"two": 2}, time.monotonic() + 1)
        left.shutdown(socket.SHUT_WR)
        assert ipc._receive_frame(right, time.monotonic() + 1) == {"one": 1}
        with pytest.raises(ipc.ProductionSignerIPCError, match="EXACTLY_ONE"):
            ipc._expect_eof(right, time.monotonic() + 1)


def test_sending_nonfinite_and_oversized_facts_rejected():
    left, right = socket.socketpair()
    with left, right:
        for value in ({"v": float("nan")}, {"v": "x" * ipc.MAX_FRAME_BYTES}):
            with pytest.raises(ipc.ProductionSignerIPCError):
                ipc._send_frame(left, value, time.monotonic() + 1)


def _child_exchange(path, ready, finish, mode):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(path)
        listener.listen(1)
        ready.set()
        peer, _ = listener.accept()
        with peer:
            value = ipc._receive_frame(peer, time.monotonic() + 2)
            ipc._expect_eof(peer, time.monotonic() + 2)
            result = response(value)
            if mode == "wrong_role":
                result["role"] = "history_signer"
            if mode != "disconnect":
                ipc._send_frame(peer, result, time.monotonic() + 2)
        finish.wait(5)


@pytest.mark.parametrize("mode", ["valid", "wrong_role", "disconnect"])
def test_real_linux_process_exchange_without_claiming_account_isolation(policy, monkeypatch, mode):
    # Both processes deliberately use the test account. Only protocol, peer PID,
    # pidfd liveness and process-start pinning are real evidence in this test.
    context = multiprocessing.get_context("fork")
    ready, finish = context.Event(), context.Event()
    child = context.Process(target=_child_exchange, args=(policy.path, ready, finish, mode))
    child.start()
    try:
        assert ready.wait(3)

        @contextmanager
        def simulated_installation(_policy):
            yield SimpleNamespace(address=policy.path, recheck=lambda: None)

        monkeypatch.setattr(ipc, "_PinnedSocketPath", simulated_installation)
        monkeypatch.setattr(ipc, "_require_self", lambda uid, gid: None)
        if mode == "valid":
            result = ipc.qualify_signer_ipc(policy)
            assert (result.peer_pid, result.peer_uid, result.peer_gid) == (
                child.pid,
                os.getuid(),
                os.getgid(),
            )
            assert result.peer_start_time > 0
            assert result.response["status"] == "BLOCKED"
        else:
            with pytest.raises(ipc.ProductionSignerIPCError):
                ipc.qualify_signer_ipc(policy)
    finally:
        finish.set()
        child.join(3)
        if child.is_alive():
            child.terminate()
            child.join(3)
        assert child.exitcode == 0


def test_same_process_socket_never_qualifies():
    left, right = socket.socketpair()
    with left, right:
        with pytest.raises(ipc.ProductionSignerIPCError, match="SEPARATE"):
            ipc._PinnedPeer(left, os.getuid(), os.getgid())


def test_wrong_peer_uid_fails_before_proc_access():
    left, right = socket.socketpair()
    with left, right:
        with pytest.raises(ipc.ProductionSignerIPCError, match="SEPARATE"):
            ipc._PinnedPeer(left, os.getuid() + 1, os.getgid())


def test_supplementary_group_rejected(monkeypatch):
    monkeypatch.setattr(os, "getgroups", lambda: [os.getgid(), os.getgid() + 1])
    with pytest.raises(ipc.ProductionSignerIPCError, match="ISOLATED_PROCESS"):
        ipc._require_self(os.getuid(), os.getgid())


def test_saved_privileged_identity_rejected(monkeypatch):
    monkeypatch.setattr(os, "getresuid", lambda: (os.getuid(), os.getuid(), 0))
    with pytest.raises(ipc.ProductionSignerIPCError, match="ISOLATED_PROCESS"):
        ipc._require_self(os.getuid(), os.getgid())


@pytest.mark.parametrize("entry", ["self", "acl", "client", "server"])
def test_non_linux_entry_points_fail_before_os_access(policy, monkeypatch, entry):
    monkeypatch.setattr(ipc.sys, "platform", "win32")
    with pytest.raises(ipc.ProductionSignerIPCError, match="LINUX_PIDFD"):
        if entry == "self":
            ipc._require_self(1000, 2000)
        elif entry == "acl":
            ipc._no_acl(-1)
        elif entry == "client":
            ipc.qualify_signer_ipc(policy)
        else:
            ipc.run_diagnostic_server(policy, lambda operation: {})


def test_actual_unprotected_path_cannot_qualify(policy):
    with pytest.raises(ipc.ProductionSignerIPCError, match="PROTECTED_SIGNER_ANCESTRY"):
        ipc._PinnedSocketPath(policy, require_socket=False)


@pytest.fixture
def simulated_metadata(policy, monkeypatch):
    """Real descriptors/inodes, simulated installation owner and modes only."""
    actual_fstat, actual_stat = os.fstat, os.stat
    parent_inode = policy.path.rsplit("/", 1)[0]
    selected_parent = actual_stat(parent_inode).st_ino

    def adjusted(info):
        values = {name: getattr(info, name) for name in dir(info) if name.startswith("st_")}
        if stat.S_ISDIR(info.st_mode):
            values["st_uid"] = 0
            values["st_mode"] &= ~0o022
            if info.st_ino == selected_parent:
                values.update(st_uid=policy.signer_uid, st_gid=policy.issuer_gid)
                values["st_mode"] = stat.S_IFDIR | 0o2710
        elif stat.S_ISSOCK(info.st_mode):
            values.update(st_uid=policy.signer_uid, st_gid=policy.issuer_gid)
            values["st_mode"] = stat.S_IFSOCK | 0o660
        return SimpleNamespace(**values)

    monkeypatch.setattr(os, "fstat", lambda fd: adjusted(actual_fstat(fd)))
    monkeypatch.setattr(os, "stat", lambda *args, **kwargs: adjusted(actual_stat(*args, **kwargs)))
    return adjusted


def test_socket_substitution_detected_with_real_inodes(policy, simulated_metadata):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as first:
        first.bind(policy.path)
        with ipc._PinnedSocketPath(policy) as pinned:
            pinned.recheck()
            os.unlink(policy.path)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as replacement:
                replacement.bind(policy.path)
                with pytest.raises(ipc.ProductionSignerIPCError, match="SOCKET_CHANGED"):
                    pinned.recheck()


def test_symlink_socket_rejected_even_with_simulated_installation(
    policy, simulated_metadata, tmp_path
):
    target = tmp_path / "other.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(target))
        os.symlink(target, policy.path)
        with pytest.raises(ipc.ProductionSignerIPCError, match="EXACT_SIGNER_SOCKET"):
            ipc._PinnedSocketPath(policy)


def test_acl_on_pinned_object_rejected(policy, simulated_metadata, monkeypatch):
    monkeypatch.setattr(os, "listxattr", lambda path: ["system.posix_acl_access"])
    with pytest.raises(ipc.ProductionSignerIPCError, match="ACL_FORBIDDEN"):
        ipc._PinnedSocketPath(policy, require_socket=False)


def test_namespace_permission_denial_does_not_invent_evidence(monkeypatch):
    start = b"123 (test process) S " + b"0 " * 18 + b"123 0\n"
    status = b"Uid:\t1000 1000 1000 1000\nGid:\t2000 2000 2000 2000\nGroups:\t2000\n"
    monkeypatch.setattr(ipc, "_read_proc", lambda fd, name: start if name == "stat" else status)

    def denied(*args, **kwargs):
        raise PermissionError

    monkeypatch.setattr(os, "stat", denied)
    assert ipc._process_identity(1, 1000, 2000) == (123, ())


def test_no_public_sign_operation(policy):
    assert ipc._OPERATIONS == {"DIAGNOSTIC", "QUALIFY"}
    with pytest.raises(ipc.ProductionSignerIPCError):
        ipc.qualify_signer_ipc(policy, "SIGN")


def test_server_rejects_sign_before_diagnostic_callback(policy, monkeypatch):
    @contextmanager
    def simulated_identity(*args):
        yield SimpleNamespace(recheck=lambda: None)

    monkeypatch.setattr(ipc, "_PinnedPeer", simulated_identity)
    left, right = socket.socketpair()
    with left, right:
        ipc._send_frame(left, request(policy, operation="SIGN"), time.monotonic() + 1)
        left.shutdown(socket.SHUT_WR)
        with pytest.raises(ipc.ProductionSignerIPCError, match="EXACT_DIAGNOSTIC"):
            ipc._serve_connection(
                right,
                policy,
                lambda op: pytest.fail("SIGN must never reach callback"),
                SimpleNamespace(recheck=lambda: None),
            )


def test_real_peer_exit_invalidates_socket_bound_pin(tmp_path):
    context = multiprocessing.get_context("fork")
    release, ready = context.Event(), context.Event()
    address = str(tmp_path / "exit.sock")

    def child_connect():
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(address)
            listener.listen(1)
            ready.set()
            peer, _ = listener.accept()
            with peer:
                release.wait(3)

    child = context.Process(target=child_connect)
    child.start()
    try:
        assert ready.wait(3)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
            peer.connect(address)
            with ipc._PinnedPeer(peer, os.getuid(), os.getgid()) as identity:
                release.set()
                child.join(3)
                assert child.exitcode == 0
                with pytest.raises(ipc.ProductionSignerIPCError, match="CHANGED_OR_EXITED"):
                    identity.recheck()
    finally:
        release.set()
        child.join(3)
