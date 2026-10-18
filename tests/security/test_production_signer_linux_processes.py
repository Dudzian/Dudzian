"""Disposable root Linux harness for real DAC and diagnostic process boundaries.

Numeric test identities and synthetic fixture files are not installed service
accounts or production credentials. These checks do not qualify systemd,
namespaces, native custody, private control or production Root Proof emission.
Run inside a disposable Linux container with its default capabilities only.
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
import socket
import stat
import sys
import tempfile
import time
from dataclasses import replace
from multiprocessing.reduction import recv_handle, send_handle
from pathlib import Path
from types import SimpleNamespace

import pytest

from bot_core import production_signer_ipc as ipc

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or os.geteuid() != 0,
    reason="disposable root Linux process harness required",
)

_ROOT_SIGNER = 61011
_HISTORY_SIGNER = 61012
_ISSUER = 61013
_WRONG_REQUESTER = 61014
_SYNTHETIC_FIXTURE = b"isolated test fixture; contains no private key\n"


def _drop_identity(uid, gid):
    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)
    assert (os.getuid(), os.geteuid(), os.getgid(), os.getegid()) == (uid, uid, gid, gid)
    assert os.getgroups() == []


def _server(policy, own_fixture, other_fixture, stop, calls, events):
    try:
        _drop_identity(policy.signer_uid, policy.signer_gid)
        assert Path(own_fixture).read_bytes() == _SYNTHETIC_FIXTURE
        try:
            Path(other_fixture).read_bytes()
        except PermissionError:
            pass
        else:
            raise AssertionError("a signer read the other signer's synthetic fixture")
        events.send(("identity", os.getpid(), os.getuid(), os.getgid(), os.getgroups()))

        def diagnostic(operation):
            with calls.get_lock():
                calls.value += 1
            return {
                "facts": {"private_control_status": "BLOCKED", "synthetic_role": policy.role},
                "blockers": ["PRIVATE_CONTROL_NOT_INDEPENDENTLY_PROVEN"],
            }

        ipc.run_diagnostic_server(policy, diagnostic, stop)
    except BaseException as error:
        events.send(("error", type(error).__name__, str(error)))
        raise
    finally:
        events.close()


def _raw_request(policy):
    request = {
        "version": 1,
        "operation": "DIAGNOSTIC",
        "role": policy.role,
        "installation_sha256": policy.installation_sha256,
        "nonce": "b" * 64,
    }
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
        peer.settimeout(2)
        peer.connect(policy.path)
        ipc._send_frame(peer, request, time.monotonic() + 2)
        peer.shutdown(socket.SHUT_WR)
        return ipc._receive_frame(peer, time.monotonic() + 2)


def _requester(uid, gid, commands, results):
    try:
        _drop_identity(uid, gid)
        results.send(("identity", os.getpid(), os.getuid(), os.getgid(), os.getgroups()))
        while True:
            command, *arguments = commands.recv()
            if command == "stop":
                return
            try:
                if command == "qualify":
                    result = ipc.qualify_signer_ipc(*arguments)
                elif command == "raw":
                    result = _raw_request(*arguments)
                elif command == "read_fixtures":
                    result = []
                    for path in arguments:
                        try:
                            Path(path).read_bytes()
                        except PermissionError:
                            result.append("PERMISSION_DENIED")
                        else:
                            result.append("READABLE")
                else:
                    raise AssertionError("unknown test command")
            except (ipc.ProductionSignerIPCError, OSError) as error:
                results.send(("denied", type(error).__name__, str(error)))
            else:
                results.send(("ok", result))
    finally:
        commands.close()
        results.close()


def _join(process):
    process.join(3)
    if process.is_alive():
        process.kill()
        process.join(3)
        raise AssertionError(f"isolated test process {process.pid} did not stop")
    assert process.exitcode == 0, f"isolated test process {process.pid} exited {process.exitcode}"


def _receive(connection):
    assert connection.poll(5), "isolated test process did not respond"
    return connection.recv()


def _start_requester(context, uid, gid):
    commands_child, commands_parent = context.Pipe(duplex=False)
    results_parent, results_child = context.Pipe(duplex=False)
    process = context.Process(target=_requester, args=(uid, gid, commands_child, results_child))
    process.start()
    commands_child.close()
    results_child.close()
    assert _receive(results_parent) == ("identity", process.pid, uid, gid, [])
    return SimpleNamespace(process=process, commands=commands_parent, results=results_parent)


def _command(requester, command, *arguments):
    requester.commands.send((command, *arguments))
    return _receive(requester.results)


def _start_server(runtime, policy):
    own, other = runtime.fixtures if policy.role == "root_signer" else runtime.fixtures[::-1]
    stop = runtime.context.Event()
    calls = runtime.context.Value("i", 0)
    events_parent, events_child = runtime.context.Pipe(duplex=False)
    process = runtime.context.Process(
        target=_server, args=(policy, own, other, stop, calls, events_child)
    )
    service = SimpleNamespace(process=process, stop=stop, calls=calls, events=events_parent)
    runtime.services[policy.role] = service
    process.start()
    events_child.close()
    assert _receive(events_parent) == (
        "identity",
        process.pid,
        policy.signer_uid,
        policy.signer_gid,
        [],
    )
    deadline = time.monotonic() + 5
    while not Path(policy.path).exists():
        assert process.is_alive(), "isolated signer exited before binding its socket"
        assert time.monotonic() < deadline, "isolated signer did not bind its socket"
        time.sleep(0.01)
    return service


@pytest.fixture(scope="module")
def linux_processes():
    # /tmp has writable ancestry and must never be accepted by the actual path
    # validator. Only this disposable harness creates test directories in /run.
    base = Path(tempfile.mkdtemp(prefix="cryptohunter-rpi02-test-", dir="/run"))
    base.chmod(0o755)
    runtime = SimpleNamespace(
        context=multiprocessing.get_context("fork"), services={}, requesters=[], fixtures=[]
    )
    try:
        policies = []
        for role, identity in (("root_signer", _ROOT_SIGNER), ("history_signer", _HISTORY_SIGNER)):
            directory = base / role
            directory.mkdir()
            os.chown(directory, identity, _ISSUER)
            directory.chmod(0o2710)
            credential_directory = base / (role + "_synthetic_fixture")
            credential_directory.mkdir(mode=0o700)
            fixture = credential_directory / "synthetic-credential.txt"
            fixture.write_bytes(_SYNTHETIC_FIXTURE)
            fixture.chmod(0o400)
            os.chown(fixture, identity, identity)
            os.chown(credential_directory, identity, identity)
            runtime.fixtures.append(str(fixture))
            policies.append(
                ipc.SignerSocketPolicy(
                    role,
                    str(directory / "signer.sock"),
                    identity,
                    identity,
                    _ISSUER,
                    _ISSUER,
                    "a" * 64,
                )
            )
        runtime.policies = tuple(policies)
        for policy in runtime.policies:
            _start_server(runtime, policy)
        runtime.issuer = _start_requester(runtime.context, _ISSUER, _ISSUER)
        runtime.requesters.append(runtime.issuer)
        yield runtime
    finally:
        cleanup_errors = []
        for requester in runtime.requesters:
            if requester.process.is_alive():
                try:
                    requester.commands.send(("stop",))
                except OSError as error:
                    cleanup_errors.append(str(error))
        for service in runtime.services.values():
            service.stop.set()
        # A child can fail during its own interpreter/coverage shutdown. Stop
        # and reap every sibling before surfacing any individual exit failure.
        for child in (*runtime.requesters, *runtime.services.values()):
            if child.process.pid is not None:
                try:
                    _join(child.process)
                except (AssertionError, OSError) as error:
                    cleanup_errors.append(str(error))
        try:
            for requester in runtime.requesters:
                requester.commands.close()
                requester.results.close()
            for service in runtime.services.values():
                service.events.close()
        finally:
            shutil.rmtree(base)
        assert not cleanup_errors, "\n".join(cleanup_errors)


def test_two_real_signers_keep_exact_socket_metadata_peer_pids_and_blocked_diagnostics(
    linux_processes,
):
    runtime = linux_processes
    observed_pids = set()
    for policy in runtime.policies:
        service = runtime.services[policy.role]
        info = os.stat(policy.path, follow_symlinks=False)
        assert stat.S_ISSOCK(info.st_mode)
        assert (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode), info.st_nlink) == (
            policy.signer_uid,
            _ISSUER,
            0o660,
            1,
        )
        directory = os.stat(Path(policy.path).parent, follow_symlinks=False)
        assert (directory.st_uid, directory.st_gid, stat.S_IMODE(directory.st_mode)) == (
            policy.signer_uid,
            _ISSUER,
            0o2710,
        )
        first, second = (_command(runtime.issuer, "qualify", policy) for _ in range(2))
        assert first[0] == second[0] == "ok", (first, second)
        first, second = first[1], second[1]
        assert first.peer_pid == second.peer_pid == service.process.pid
        assert first.peer_start_time == second.peer_start_time > 0
        assert (first.peer_uid, first.peer_gid) == (policy.signer_uid, policy.signer_gid)
        assert first.response["nonce"] != second.response["nonce"]
        for result in (first, second):
            assert result.response["role"] == policy.role
            assert result.response["status"] == result.response["production_emission"] == "BLOCKED"
            assert result.response["facts"]["private_control_status"] == "BLOCKED"
            assert result.response["blockers"] == ["PRIVATE_CONTROL_NOT_INDEPENDENTLY_PROVEN"]
        observed_pids.add(first.peer_pid)
    assert len(observed_pids | {runtime.issuer.process.pid}) == 3


def test_real_issuer_cannot_read_either_signers_synthetic_credential_fixture(linux_processes):
    runtime = linux_processes
    assert _command(runtime.issuer, "read_fixtures", *runtime.fixtures) == (
        "ok",
        ["PERMISSION_DENIED", "PERMISSION_DENIED"],
    )


def test_wrong_requester_kernel_uid_is_denied_before_diagnostic_callback(linux_processes):
    runtime = linux_processes
    # The issuer GID grants socket traversal, so this reaches the server rather
    # than relying only on pathname DAC. SO_PEERCRED must reject the wrong UID.
    wrong = _start_requester(runtime.context, _WRONG_REQUESTER, _ISSUER)
    runtime.requesters.append(wrong)
    for policy in runtime.policies:
        before = runtime.services[policy.role].calls.value
        assert _command(wrong, "raw", policy)[0] == "denied"
        assert runtime.services[policy.role].calls.value == before
        assert _command(runtime.issuer, "qualify", policy)[0] == "ok"


def test_wrong_role_socket_or_request_cannot_substitute_for_installed_signer(linux_processes):
    runtime = linux_processes
    root, history = runtime.policies
    before = runtime.services[history.role].calls.value
    wrong_path = replace(root, path=history.path)
    assert _command(runtime.issuer, "qualify", wrong_path)[0] == "denied"
    # Exact history endpoint metadata passes, then its server rejects root's
    # semantic role before any diagnostic callback executes.
    wrong_role = replace(history, role=root.role)
    assert _command(runtime.issuer, "qualify", wrong_role)[0] == "denied"
    assert runtime.services[history.role].calls.value == before
    assert _command(runtime.issuer, "qualify", root)[0] == "ok"
    assert _command(runtime.issuer, "qualify", history)[0] == "ok"


def test_graceful_signer_restart_removes_socket_and_observes_new_process(linux_processes):
    runtime = linux_processes
    policy = runtime.policies[0]
    original = runtime.services[policy.role]
    result = _command(runtime.issuer, "qualify", policy)
    assert result[0] == "ok"
    original.stop.set()
    _join(original.process)
    original.events.close()
    assert not Path(policy.path).exists()
    replacement = _start_server(runtime, policy)
    assert replacement.process.pid != original.process.pid
    result = _command(runtime.issuer, "qualify", policy)
    assert result[0] == "ok"
    assert result[1].peer_pid == replacement.process.pid
    assert result[1].response["facts"]["private_control_status"] == "BLOCKED"


def _accepted_socket_origin(policy, control):
    _drop_identity(policy.signer_uid, policy.signer_gid)
    try:
        with ipc._PinnedSocketPath(policy, require_socket=False) as path:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                old_umask = os.umask(0o117)
                try:
                    listener.bind(path.address)
                finally:
                    os.umask(old_umask)
                path.pin_socket()
                listener.listen(1)
                listener.settimeout(3)
                control.send(("ready", os.getpid()))
                peer, _ = listener.accept()
                with peer:
                    # The parent retains this connection after the originating
                    # signer exits. An open inherited socket is not liveness.
                    send_handle(control, peer.fileno(), os.getppid())
    finally:
        control.close()


def test_dead_signer_origin_is_rejected_while_transferred_socket_remains_open(linux_processes):
    runtime = linux_processes
    policy = replace(
        runtime.policies[0],
        path=str(Path(runtime.policies[0].path).with_name("dead-origin.sock")),
    )
    parent, child = runtime.context.Pipe()
    origin = runtime.context.Process(target=_accepted_socket_origin, args=(policy, child))
    origin.start()
    child.close()
    try:
        assert _receive(parent) == ("ready", origin.pid)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(3)
            client.connect(policy.path)
            assert parent.poll(3), "origin did not transfer its accepted socket"
            with socket.socket(fileno=recv_handle(parent)) as retained:
                _join(origin)
                assert retained.fileno() >= 0
                with pytest.raises((ipc.ProductionSignerIPCError, OSError)):
                    ipc._PinnedPeer(client, policy.signer_uid, policy.signer_gid)
    finally:
        if origin.is_alive():
            origin.kill()
            origin.join(3)
        parent.close()
        Path(policy.path).unlink(missing_ok=True)
