"""Prawdziwy rootless AF_UNIX/PIDFD oraz jawne modele chronionej instalacji.

To nie jest dowód działania zainstalowanej usługi ani izolacji obcych UID.
NSS/runtime UID, root-owned ancestry i systemd cgroup są modelowane; socket,
niezależny proces, peer credentials, PIDFD, procfs i ramki są rzeczywiste.
Writer bazy jest modelem obserwującym wywołania, bez połączeń PostgreSQL.
"""

from __future__ import annotations

import base64
import hashlib
import multiprocessing
import os
import queue
import socket
import stat
import struct
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from bot_core import freshness_runtime_ipc as ipc
from bot_core.freshness_semantic_verifier import (
    RECEIPT_AUTHENTICATION_DOMAIN,
    serve_local_unix_socket,
)
from bot_core.linux_service_ipc import capture_process_identity
from bot_core.postgresql_freshness_authority import (
    canonical_json_bytes,
    complete_semantic_head_digest,
    parse_canonical_json,
)
from bot_core.production_signer_ipc import _SO_PEERPIDFD
from deployment import linux_physical_ipc as paths
from tests.security.test_freshness_semantic_verifier import fixture_candidate

LINUX = pytest.mark.skipif(sys.platform != "linux", reason="Wymagane Linux AF_UNIX i PIDFD")


def _frame(raw):
    return struct.pack("!I", len(raw)) + raw


def _read_exact(peer, count):
    result = bytearray()
    while len(result) < count:
        part = peer.recv(count - len(result))
        if not part:
            return bytes(result)
        result.extend(part)
    return bytes(result)


def _record_expected(record):
    return paths.ExpectedPeerProcess(
        record["pid"],
        record["uid"],
        record["gid"],
        record["start_time_ticks"],
        record["exe_sha256"],
    )


def _bind_private_test_socket(listener, address):
    """Utwórz atomowo socket 0660 wyłącznie w kontrolowanym procesie potomnym.

    Umask jest globalna dla procesu. Główny pytest może mieć wątki, więc
    ta fixture nigdy nie zmienia jego umask. Prywatny katalog 0700 ogranicza
    dostęp do testowego UID; 0660 odtwarza dokładny kontrakt produkcji.
    """
    if multiprocessing.current_process().name == "MainProcess":
        raise RuntimeError("Bind fixture wymaga odrębnego procesu")
    directory = Path(address).parent
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parent = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(parent.st_mode)
            or (parent.st_uid, parent.st_gid) != (os.geteuid(), os.getegid())
            or stat.S_IMODE(parent.st_mode) != 0o700
            or {"system.posix_acl_access", "system.posix_acl_default"}.intersection(
                os.listxattr(descriptor)
            )
        ):
            raise PermissionError("Fixture wymaga prywatnego katalogu 0700 bez ACL")
        # Socket powstaje od razu z docelowym DAC; nie ma bind -> chmod
        # ani zmiany uprawnień przez ponownie rozwiązywaną ścieżkę.
        previous = os.umask(0o117)
        try:
            listener.bind(f"/proc/self/fd/{descriptor}/{Path(address).name}")
        finally:
            os.umask(previous)
        info = os.stat(Path(address).name, dir_fd=descriptor, follow_symlinks=False)
        if (
            not stat.S_ISSOCK(info.st_mode)
            or (info.st_uid, info.st_gid) != (os.geteuid(), os.getegid())
            or stat.S_IMODE(info.st_mode) != 0o660
            or info.st_nlink != 1
            or {"system.posix_acl_access", "system.posix_acl_default"}.intersection(
                os.listxattr(address, follow_symlinks=False)
            )
        ):
            raise PermissionError("Fixture nie utworzyła dokładnego socketu AF_UNIX")
    finally:
        os.close(descriptor)


@pytest.fixture
def kernel_pidfd():
    if sys.platform != "linux":
        pytest.skip("BLOCKED: kernelowe SO_PEERPIDFD wymaga Linux")
    left, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    with left, peer:
        # Oddzielny test modelu odmowy potwierdza brak fallbacku do pidfd_open.
        try:
            descriptor = peer.getsockopt(socket.SOL_SOCKET, _SO_PEERPIDFD)
        except OSError as exc:
            pytest.skip(f"BLOCKED: SO_PEERPIDFD niedostępny, errno={exc.errno}")
        else:
            os.close(descriptor)


def _scripted_server(address, channel, finish, response, mode):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        _bind_private_test_socket(listener, address)
        listener.listen(4)
        listener.settimeout(0.05)
        channel.send(capture_process_identity(os.getpid()))
        while True:
            try:
                peer, _ = listener.accept()
                break
            except TimeoutError:
                if finish.is_set():
                    channel.send(b"")
                    channel.send(0)
                    return
        with peer:
            peer.settimeout(5)
            header = _read_exact(peer, 4)
            raw = b""
            if len(header) == 4:
                raw = _read_exact(peer, struct.unpack("!I", header)[0])
            channel.send(raw)
            if mode in {"exec", "exec_same"} and raw:
                descriptor = peer.fileno()
                os.set_inheritable(descriptor, True)
                executable = sys.executable if mode == "exec_same" else "/usr/bin/python3"
                os.execv(
                    executable,
                    [
                        executable,
                        "-I",
                        "-s",
                        "-B",
                        "-c",
                        "import socket,sys,time;"
                        "p=socket.socket(fileno=int(sys.argv[1]));"
                        "p.sendall(bytes.fromhex(sys.argv[2]));"
                        "p.shutdown(socket.SHUT_WR);time.sleep(5)",
                        str(descriptor),
                        response.hex(),
                    ],
                )
            replacement = None
            if mode == "replace" and raw:
                os.unlink(address)
                replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                _bind_private_test_socket(replacement, address)
            if mode != "timeout" and raw:
                try:
                    peer.sendall(response)
                    peer.shutdown(socket.SHUT_WR)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            finish.wait(5)
            if replacement is not None:
                replacement.close()
        # Każda automatyczna ponowna próba pozostaje w rzeczywistym backlogu.
        listener.settimeout(0.05)
        connections = 1
        try:
            while True:
                extra, _ = listener.accept()
                connections += 1
                extra.close()
        except TimeoutError:
            pass
        channel.send(connections)


@contextmanager
def _scripted(tmp_path, response=None, *, mode="normal"):
    ctx = multiprocessing.get_context("fork")
    channel, child = ctx.Pipe()
    finish = ctx.Event()
    address = str(tmp_path / "verifier.sock")
    response = _frame(b'{"outcome":"INVALID"}') if response is None else response
    process = ctx.Process(target=_scripted_server, args=(address, child, finish, response, mode))
    process.start()
    child.close()
    try:
        assert channel.poll(5), "Brak gotowości niezależnego procesu"
        record = channel.recv()
        yield SimpleNamespace(
            address=address,
            expected=_record_expected(record),
            record=record,
            channel=channel,
            process=process,
            finish=finish,
        )
    finally:
        finish.set()
        process.join(1)
        if process.is_alive():
            process.terminate()
            process.join(5)
        channel.close()
        assert not process.is_alive()


def _model_installed_coordinates(monkeypatch, server):
    """Tylko model instalacji; nie zastępujemy SO_PEERCRED/PIDFD ani procfs."""
    monkeypatch.setattr(ipc, "VERIFIER_SOCKET_PATH", server.address)
    monkeypatch.setattr(ipc, "CGROUP", server.record["cgroup"])
    monkeypatch.setattr(ipc, "_require_runtime_identity", lambda expected: None)
    original_os = paths.os
    parent = str(Path(server.address).parent)

    def modeled_fstat(descriptor):
        info = original_os.fstat(descriptor)
        # Rootless runner nie jest root. Wszystkie rzeczywiste katalogi i
        # O_NOFOLLOW zostają; modelujemy wyłącznie właściciela/przepisy DAC
        # przodków powyżej dedykowanego katalogu usługi.
        if stat.S_ISDIR(info.st_mode):
            actual = original_os.readlink(f"/proc/self/fd/{descriptor}")
            if actual != parent:
                fields = list(info)
                fields[0] &= ~0o022
                fields[4] = 0
                return original_os.stat_result(fields)
        return info

    monkeypatch.setattr(
        paths,
        "os",
        SimpleNamespace(
            **{
                name: modeled_fstat if name == "fstat" else getattr(original_os, name)
                for name in dir(original_os)
            }
        ),
    )


def _finish_count(server, connections=1):
    server.finish.set()
    assert server.channel.poll(5)
    request = server.channel.recv()
    server.finish.set()
    assert server.channel.poll(5)
    assert server.channel.recv() == connections, "Niezgodna liczba połączeń lub automatyczne retry"
    return request


@LINUX
def test_real_local_unix_independent_process_exact_request_and_invalid(
    tmp_path, monkeypatch, kernel_pidfd
):
    with _scripted(tmp_path) as server:
        _model_installed_coordinates(monkeypatch, server)
        result = ipc.RuntimeFreshnessVerifierClient(server.expected).verify_exact_candidate(
            b'{"p":1}', b'{"d":2}', b'{"r":3}'
        )
        assert result.outcome is ipc.FreshnessIPCOutcome.INVALID
        request = parse_canonical_json(_finish_count(server))
        assert request == {
            "proposer_authentication": base64.urlsafe_b64encode(b'{"p":1}').rstrip(b"=").decode(),
            "authoritative_document": base64.urlsafe_b64encode(b'{"d":2}').rstrip(b"=").decode(),
            "finalization_receipt": base64.urlsafe_b64encode(b'{"r":3}').rstrip(b"=").decode(),
        }
        assert result.preparation_id is None and result.binding is None


@pytest.mark.parametrize("outcome", ["INVALID", "UNAVAILABLE", "OUTCOME_UNKNOWN"])
@LINUX
def test_real_explicit_outcomes_without_automatic_retry(
    tmp_path, monkeypatch, kernel_pidfd, outcome
):
    with _scripted(tmp_path, _frame(canonical_json_bytes({"outcome": outcome}))) as server:
        _model_installed_coordinates(monkeypatch, server)
        result = ipc.RuntimeFreshnessVerifierClient(server.expected).verify_exact_candidate(
            b"{}", b"{}", b"{}"
        )
        assert result.outcome is ipc.FreshnessIPCOutcome[outcome]
        assert _finish_count(server)


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b"[]",
        b"null",
        b"\xff",
        b"{",
        b'{"outcome": "INVALID"}',
        b'{"outcome":"INVALID","outcome":"INVALID"}',
        b'{"outcome":NaN}',
        b'{"outcome":1e999}',
        b'{"outcome":"PASS"}',
        b'{"outcome":"INVALID","extra":true}',
        b'{"outcome":true}',
        b'{"outcome":"PREPARED","binding":"=","preparation_id":"bad"}',
    ],
)
@LINUX
def test_real_malformed_json_or_response_is_ambiguous_without_retry(
    tmp_path, monkeypatch, kernel_pidfd, raw
):
    with _scripted(tmp_path, _frame(raw)) as server:
        _model_installed_coordinates(monkeypatch, server)
        result = ipc.RuntimeFreshnessVerifierClient(server.expected).verify_exact_candidate(
            b"{}", b"{}", b"{}"
        )
        assert result.outcome is ipc.FreshnessIPCOutcome.OUTCOME_UNKNOWN
        assert _finish_count(server)


@pytest.mark.parametrize(
    "response",
    [
        b"",
        b"\0\0",
        struct.pack("!I", 0),
        struct.pack("!I", 1) + b"{",
        struct.pack("!I", 4 * 1024 * 1024 + 1),
        struct.pack("!I", 2**32 - 1),
        struct.pack("!I", 100) + b"{}",
        _frame(b'{"outcome":"INVALID"}') + b"x",
        _frame(b'{"outcome":"INVALID"}') * 2,
    ],
)
@LINUX
def test_real_bad_length_truncation_or_extra_frame_never_retries(
    tmp_path, monkeypatch, kernel_pidfd, response
):
    with _scripted(tmp_path, response) as server:
        _model_installed_coordinates(monkeypatch, server)
        result = ipc.RuntimeFreshnessVerifierClient(server.expected).verify_exact_candidate(
            b"{}", b"{}", b"{}"
        )
        assert result.outcome is ipc.FreshnessIPCOutcome.OUTCOME_UNKNOWN
        assert _finish_count(server)


@LINUX
def test_real_timeout_is_bounded_and_unknown_without_retry(tmp_path, monkeypatch, kernel_pidfd):
    with _scripted(tmp_path, mode="timeout") as server:
        _model_installed_coordinates(monkeypatch, server)
        monkeypatch.setattr(ipc, "EXCHANGE_TIMEOUT_SECONDS", 0.5)
        start = time.monotonic()
        result = ipc.RuntimeFreshnessVerifierClient(server.expected).verify_exact_candidate(
            b"{}", b"{}", b"{}"
        )
        assert time.monotonic() - start < 2
        assert result.outcome is ipc.FreshnessIPCOutcome.OUTCOME_UNKNOWN
        assert _finish_count(server)


@pytest.mark.parametrize(
    "changes",
    [
        {"uid": 100000},
        {"gid": 100000},
        {"pid": 100000},
        {"start_time_ticks": 1},
        {"executable_sha256": "0" * 64},
    ],
)
@LINUX
def test_real_foreign_identity_or_mismatched_process_denied_before_send(
    tmp_path, monkeypatch, kernel_pidfd, changes
):
    with _scripted(tmp_path) as server:
        _model_installed_coordinates(monkeypatch, server)
        expected = replace(server.expected, **changes)
        result = ipc.RuntimeFreshnessVerifierClient(expected).verify_exact_candidate(
            b"{}", b"{}", b"{}"
        )
        assert result.outcome is ipc.FreshnessIPCOutcome.UNAVAILABLE
        assert _finish_count(server, connections=0 if {"uid", "gid"} & changes.keys() else 1) == b""


@LINUX
def test_real_wrong_cgroup_denied_before_send(tmp_path, monkeypatch, kernel_pidfd):
    with _scripted(tmp_path) as server:
        _model_installed_coordinates(monkeypatch, server)
        monkeypatch.setattr(ipc, "CGROUP", "/system.slice/incompatible.service")
        result = ipc.RuntimeFreshnessVerifierClient(server.expected).verify_exact_candidate(
            b"{}", b"{}", b"{}"
        )
        assert result.outcome is ipc.FreshnessIPCOutcome.UNAVAILABLE
        assert _finish_count(server) == b""


@LINUX
def test_real_socket_replacement_after_request_is_unknown(tmp_path, monkeypatch, kernel_pidfd):
    with _scripted(tmp_path, mode="replace") as server:
        _model_installed_coordinates(monkeypatch, server)
        result = ipc.RuntimeFreshnessVerifierClient(server.expected).verify_exact_candidate(
            b"{}", b"{}", b"{}"
        )
        assert result.outcome is ipc.FreshnessIPCOutcome.OUTCOME_UNKNOWN
        assert _finish_count(server)


@pytest.mark.parametrize("mode", ["exec", "exec_same"])
@LINUX
def test_real_process_exec_identity_change_after_request_is_unknown(
    tmp_path, monkeypatch, kernel_pidfd, mode
):
    if mode == "exec" and not Path("/usr/bin/python3").exists():
        pytest.skip("BLOCKED: brak odrębnego interpretera do rzeczywistego exec")
    other_hash = hashlib.sha256(
        Path(sys.executable if mode == "exec_same" else "/usr/bin/python3").read_bytes()
    ).hexdigest()
    with _scripted(tmp_path, mode=mode) as server:
        if mode == "exec" and other_hash == server.expected.executable_sha256:
            pytest.skip("BLOCKED: alternatywny interpreter nie zmienia programu procesu")
        _model_installed_coordinates(monkeypatch, server)
        result = ipc.RuntimeFreshnessVerifierClient(server.expected).verify_exact_candidate(
            b"{}", b"{}", b"{}"
        )
        assert result.outcome is ipc.FreshnessIPCOutcome.OUTCOME_UNKNOWN
        assert server.channel.poll(5) and server.channel.recv()
        changed = capture_process_identity(server.expected.pid)
        assert changed["start_time_ticks"] == server.expected.start_time_ticks
        assert changed["exe_sha256"] == other_hash


@pytest.mark.parametrize("kind", ["socket_symlink", "ancestor_symlink", "regular_file"])
@LINUX
def test_real_symlink_and_non_socket_refused_before_connect(tmp_path, monkeypatch, kind):
    directory = tmp_path / "dedicated"
    directory.mkdir(mode=0o700)
    actual = directory / "verifier.sock"
    with _scripted(directory) as server:
        _model_installed_coordinates(monkeypatch, server)
        if kind == "socket_symlink":
            selected = directory / "alias.sock"
            selected.symlink_to(actual)
        elif kind == "ancestor_symlink":
            link = tmp_path / "alias"
            link.symlink_to(directory, target_is_directory=True)
            selected = link / actual.name
        else:
            selected = directory / "regular.sock"
            selected.write_bytes(b"not a socket")
        monkeypatch.setattr(ipc, "VERIFIER_SOCKET_PATH", str(selected))
        result = ipc.RuntimeFreshnessVerifierClient(server.expected).verify_exact_candidate(
            b"{}", b"{}", b"{}"
        )
        assert result.outcome is ipc.FreshnessIPCOutcome.UNAVAILABLE
        assert _finish_count(server, connections=0) == b""


class _ObservedWriter:
    """Model braku/obecności przygotowania; nigdy nie otwiera bazy."""

    def __init__(self, events):
        self.events = events

    def prepare(self, binding, _document, _receipt):
        self.events.put(binding)
        return parse_canonical_json(binding)["preparation_id"]


def _run_semantic(verifier, address, channel, events, allowed_uid):
    verifier = replace(verifier, preparations=_ObservedWriter(events))
    channel.send(capture_process_identity(os.getpid()))
    serve_local_unix_socket(verifier, address, allowed_peer_uid=allowed_uid)


@contextmanager
def _semantic(tmp_path, *, deny_uid=False):
    ctx = multiprocessing.get_context("fork")
    verifier, _writer, proposer, document, receipt, *_ = fixture_candidate()
    inputs = tuple(canonical_json_bytes(value) for value in (proposer, document, receipt))
    channel, child = ctx.Pipe()
    events = ctx.Queue()
    address = str(tmp_path / "semantic.sock")
    process = ctx.Process(
        target=_run_semantic,
        args=(
            verifier,
            address,
            child,
            events,
            os.getuid() + 10000 if deny_uid else os.getuid(),
        ),
    )
    process.start()
    child.close()
    try:
        assert channel.poll(5)
        record = channel.recv()
        deadline = time.monotonic() + 5
        while not Path(address).exists():
            assert time.monotonic() < deadline and process.is_alive()
            time.sleep(0.005)
        yield SimpleNamespace(
            address=address,
            expected=_record_expected(record),
            record=record,
            inputs=inputs,
            events=events,
        )
    finally:
        process.terminate()
        process.join(5)
        assert not process.is_alive()
        channel.close()
        events.close()
        events.join_thread()


@LINUX
def test_real_semantic_service_and_three_signatures_prepare_once(
    tmp_path, monkeypatch, kernel_pidfd
):
    with _semantic(tmp_path) as server:
        _model_installed_coordinates(monkeypatch, server)
        result = ipc.RuntimeFreshnessVerifierClient(server.expected).verify_exact_candidate(
            *server.inputs
        )
        assert result.outcome is ipc.FreshnessIPCOutcome.PREPARED
        binding = server.events.get(timeout=2)
        assert result.binding == binding
        assert result.preparation_id == parse_canonical_json(binding)["preparation_id"]
        with pytest.raises(queue.Empty):
            server.events.get(timeout=0.1)


@LINUX
def test_real_semantic_invalid_signature_has_no_preparation(tmp_path, monkeypatch, kernel_pidfd):
    with _semantic(tmp_path) as server:
        _model_installed_coordinates(monkeypatch, server)
        proposer = parse_canonical_json(server.inputs[0])
        proposer["authentication_tag_or_signature"] = "A" * 86
        result = ipc.RuntimeFreshnessVerifierClient(server.expected).verify_exact_candidate(
            canonical_json_bytes(proposer), *server.inputs[1:]
        )
        assert result.outcome is ipc.FreshnessIPCOutcome.INVALID
        with pytest.raises(queue.Empty):
            server.events.get(timeout=0.1)


@LINUX
def test_real_server_denies_unauthorized_runtime_before_preparation(
    tmp_path, monkeypatch, kernel_pidfd
):
    with _semantic(tmp_path, deny_uid=True) as server:
        _model_installed_coordinates(monkeypatch, server)
        result = ipc.RuntimeFreshnessVerifierClient(server.expected).verify_exact_candidate(
            *server.inputs
        )
        assert result.outcome is ipc.FreshnessIPCOutcome.OUTCOME_UNKNOWN
        with pytest.raises(queue.Empty):
            server.events.get(timeout=0.1)


def test_unsupported_platform_refuses_before_socket_or_nss(monkeypatch):
    expected = paths.ExpectedPeerProcess(123, 1000, 1000, 1, "a" * 64)
    monkeypatch.setattr(ipc, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(
        ipc,
        "socket",
        SimpleNamespace(socket=lambda *_a: pytest.fail("Nie wolno otwierać transportu")),
    )
    monkeypatch.setattr(
        ipc, "_require_runtime_identity", lambda *_a: pytest.fail("Nie wolno czytać NSS")
    )
    result = ipc.RuntimeFreshnessVerifierClient(expected).verify_exact_candidate(
        b"{}", b"{}", b"{}"
    )
    assert result.outcome is ipc.FreshnessIPCOutcome.UNAVAILABLE


@LINUX
def test_model_pidfd_unavailable_fails_closed_without_fallback(tmp_path, monkeypatch):
    """Model brakującego syscallu; rzeczywisty transport nie dostaje PASS."""
    with _scripted(tmp_path) as server:
        _model_installed_coordinates(monkeypatch, server)
        original_peer = ipc.PinnedReadOnlyPeer

        def unavailable(*_args, **_kwargs):
            raise OSError(92, "Model braku SO_PEERPIDFD")

        monkeypatch.setattr(ipc, "PinnedReadOnlyPeer", unavailable)
        result = ipc.RuntimeFreshnessVerifierClient(server.expected).verify_exact_candidate(
            b"{}", b"{}", b"{}"
        )
        assert result.outcome is ipc.FreshnessIPCOutcome.UNAVAILABLE
        assert _finish_count(server) == b""
        assert original_peer is not unavailable


@pytest.mark.parametrize("field", ["uid", "gid"])
@LINUX
def test_kernel_peer_credentials_reject_foreign_expected_identity(
    tmp_path, monkeypatch, kernel_pidfd, field
):
    """Model katalogu izoluje rzeczywistą jądrową odmowę SO_PEERCRED."""
    with _scripted(tmp_path) as server:
        _model_installed_coordinates(monkeypatch, server)
        original_path_guard = ipc._pinned_verifier_socket
        monkeypatch.setattr(
            ipc,
            "_pinned_verifier_socket",
            lambda address, _expected: original_path_guard(address, server.expected),
        )
        expected = replace(server.expected, **{field: getattr(server.expected, field) + 10000})
        result = ipc.RuntimeFreshnessVerifierClient(expected).verify_exact_candidate(
            b"{}", b"{}", b"{}"
        )
        assert result.outcome is ipc.FreshnessIPCOutcome.UNAVAILABLE
        assert _finish_count(server) == b""


def _fixture_socket_contract_child(directory, channel, case):
    root = Path(directory)
    if case == "unsafe_parent":
        root = root / "group-visible"
        directory_umask = os.umask(0o027)
        try:
            root.mkdir(mode=0o750)
        finally:
            os.umask(directory_umask)
    elif case == "parent_symlink":
        target = root / "real"
        target.mkdir(mode=0o700)
        root = root / "alias"
        root.symlink_to(target, target_is_directory=True)
    address = str(root / "contract.sock")
    if case == "bind_failure":
        Path(address).write_bytes(b"istniejacy obiekt fixture")

    class RegularFileInsteadOfSocket:
        def bind(self, selected):
            Path(selected).write_bytes(b"zly typ obiektu fixture")

    previous = os.umask(0o777)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            try:
                _bind_private_test_socket(
                    RegularFileInsteadOfSocket() if case == "wrong_type" else listener, address
                )
            except (OSError, ValueError) as exc:
                outcome = type(exc).__name__
            else:
                outcome = "BOUND"
            restored = os.umask(0o777)
            info = Path(address).lstat() if Path(address).exists() else None
            channel.send(
                {
                    "outcome": outcome,
                    "restored_umask": restored,
                    "socket": info is not None and stat.S_ISSOCK(info.st_mode),
                    "mode": None if info is None else stat.S_IMODE(info.st_mode),
                    "uid": None if info is None else info.st_uid,
                    "gid": None if info is None else info.st_gid,
                    "links": None if info is None else info.st_nlink,
                }
            )
    finally:
        os.umask(previous)
        channel.close()


@LINUX
@pytest.mark.parametrize(
    "case", ["success", "bind_failure", "unsafe_parent", "parent_symlink", "wrong_type"]
)
def test_real_fixture_socket_dac_and_umask_restore(tmp_path, case):
    """Prawdziwe DAC i typ inode; umask zmienia tylko jednorazowy potomek."""
    context = multiprocessing.get_context("fork")
    channel, child = context.Pipe()
    process = context.Process(
        target=_fixture_socket_contract_child, args=(str(tmp_path), child, case)
    )
    process.start()
    child.close()
    try:
        assert channel.poll(5)
        result = channel.recv()
        assert result["restored_umask"] == 0o777
        if case == "success":
            assert result == {
                "outcome": "BOUND",
                "restored_umask": 0o777,
                "socket": True,
                "mode": 0o660,
                "uid": os.geteuid(),
                "gid": os.getegid(),
                "links": 1,
            }
        else:
            assert result["outcome"] != "BOUND"
        process.join(5)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        channel.close()


@LINUX
def test_fixture_socket_bind_refuses_to_change_main_process_umask(tmp_path):
    with pytest.raises(RuntimeError, match="odrębnego procesu"):
        _bind_private_test_socket(object(), str(tmp_path / "forbidden.sock"))
    assert not (tmp_path / "forbidden.sock").exists()


class _DerivedIdWriter:
    """Model PostgreSQL użyty wyłącznie do wytworzenia poprawnego bindingu."""

    def prepare(self, binding, _document, _receipt):
        return parse_canonical_json(binding)["preparation_id"]


def _valid_binding():
    verifier, _writer, proposer, document, receipt, *_ = fixture_candidate()
    verifier = replace(verifier, preparations=_DerivedIdWriter())
    inputs = tuple(canonical_json_bytes(value) for value in (proposer, document, receipt))
    return parse_canonical_json(verifier.verify_exact_candidate(*inputs).binding)


def _prepared_response(binding, *, rehash=True):
    binding = dict(binding)
    if rehash:
        binding["preparation_id"] = hashlib.sha256(
            ipc.PREPARATION_DOMAIN
            + canonical_json_bytes(
                {key: value for key, value in binding.items() if key != "preparation_id"}
            )
        ).hexdigest()
    return canonical_json_bytes(
        {
            "outcome": "PREPARED",
            "preparation_id": binding["preparation_id"],
            "binding": base64.urlsafe_b64encode(canonical_json_bytes(binding))
            .rstrip(b"=")
            .decode(),
        }
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_predecessor_generation", True),
        ("expected_predecessor_generation", {}),
        ("proposer_key_version", 0),
        ("finalization_key_version", False),
        ("proposer_lifecycle_generation_observed_for_key_binding_only", None),
        ("finalization_lifecycle_generation_observed_for_key_binding_only", -1),
        ("proposer_identity", ""),
        ("trust_domain", {}),
        ("authority_id", None),
        ("receipt_id", None),
        ("finalization_request_id", ""),
        ("proposer_credential_role_identity", ""),
        ("finalization_credential_role_identity", 1),
        ("expected_predecessor_document_digest", "A" * 64),
        ("expected_predecessor_complete_semantic_head_digest", None),
        ("proposed_document_digest", "0"),
        ("original_decision_identity", {}),
        ("proposer_public_key_material_identity", "bad"),
        ("proposer_authentication_digest", ""),
        ("finalization_public_key_material_identity", 7),
        ("authoritative_document_authentication_digest", False),
        ("receipt_canonical_digest", []),
        ("schema_version", True),
        ("schema_version", 2),
        ("security_profile", "TEST_ONLY"),
        ("environment", "TEST"),
        ("operation_type", "SIGN"),
        ("verifier_authority_identity", "unrelated"),
        ("verifier_authority_version", True),
    ],
)
def test_model_prepared_binding_shape_is_rejected_even_with_correct_digest(field, value):
    """Prawdziwy parser i hash; modelowany issuer bez bazy i bez IPC."""
    binding = _valid_binding()
    binding[field] = value
    with pytest.raises(ValueError):
        ipc._decode_response(_prepared_response(binding))


@pytest.mark.parametrize(
    "mutation", ["digest", "missing_field", "extra_field", "binding_padding", "response_id"]
)
def test_model_prepared_binding_exact_identity_and_encoding(mutation):
    binding = _valid_binding()
    if mutation == "digest":
        binding["preparation_id"] = "0" * 64
        raw = _prepared_response(binding, rehash=False)
    elif mutation == "missing_field":
        del binding["authority_id"]
        raw = _prepared_response(binding)
    elif mutation == "extra_field":
        binding["extra"] = True
        raw = _prepared_response(binding)
    else:
        value = parse_canonical_json(_prepared_response(binding))
        if mutation == "binding_padding":
            value["binding"] += "="
        else:
            value["preparation_id"] = None
        raw = canonical_json_bytes(value)
    with pytest.raises(ValueError):
        ipc._decode_response(raw)


@pytest.mark.parametrize(
    "inputs",
    [
        (None, b"{}", b"{}"),
        (b"", b"{}", b"{}"),
        (bytearray(b"{}"), b"{}", b"{}"),
        (b"x" * (4 * 1024 * 1024 + 1), b"{}", b"{}"),
        (b"x" * (3 * 1024 * 1024), b"{}", b"{}"),
    ],
)
@LINUX
def test_invalid_or_oversized_local_input_never_connects(monkeypatch, inputs):
    expected = paths.ExpectedPeerProcess(123, 1000, 1000, 1, "a" * 64)
    monkeypatch.setattr(ipc, "_require_runtime_identity", lambda _expected: None)
    monkeypatch.setattr(
        ipc, "_pinned_verifier_socket", lambda *_args: pytest.fail("Błąd musi poprzedzać connect")
    )
    result = ipc.RuntimeFreshnessVerifierClient(expected).verify_exact_candidate(*inputs)
    assert result.outcome is ipc.FreshnessIPCOutcome.INVALID


def test_no_public_socket_path_or_transport_selection():
    with pytest.raises(TypeError):
        ipc.RuntimeFreshnessVerifierClient("/tmp/caller-selected.sock")
    with pytest.raises(TypeError):
        ipc.RuntimeFreshnessVerifierClient(None, socket_path="/tmp/caller-selected.sock")


@pytest.mark.parametrize("when", [2, 3])
@LINUX
def test_model_runtime_identity_drift_around_real_exchange_never_retries(
    tmp_path, monkeypatch, kernel_pidfd, when
):
    with _scripted(tmp_path) as server:
        _model_installed_coordinates(monkeypatch, server)
        calls = 0

        def changed(_expected):
            nonlocal calls
            calls += 1
            if calls == when:
                raise PermissionError("Model zmiany credentials runtime")

        monkeypatch.setattr(ipc, "_require_runtime_identity", changed)
        result = ipc.RuntimeFreshnessVerifierClient(server.expected).verify_exact_candidate(
            b"{}", b"{}", b"{}"
        )
        expected = (
            ipc.FreshnessIPCOutcome.UNAVAILABLE
            if when == 2
            else ipc.FreshnessIPCOutcome.OUTCOME_UNKNOWN
        )
        assert result.outcome is expected
        request = _finish_count(server)
        assert bool(request) == (when == 3)


@LINUX
def test_real_self_hashed_malformed_binding_remains_unknown(tmp_path, monkeypatch, kernel_pidfd):
    binding = _valid_binding()
    binding["expected_predecessor_generation"] = {}
    with _scripted(tmp_path, _frame(_prepared_response(binding))) as server:
        _model_installed_coordinates(monkeypatch, server)
        result = ipc.RuntimeFreshnessVerifierClient(server.expected).verify_exact_candidate(
            b"{}", b"{}", b"{}"
        )
        assert result.outcome is ipc.FreshnessIPCOutcome.OUTCOME_UNKNOWN
        assert _finish_count(server)


def _two_verified_candidates():
    """Dwa poprawne kandydaty, jeden resolver, rzeczywiste podpisy; writer jest modelem."""
    verifier, _writer, proposer, document, receipt, pkey, fkey = fixture_candidate()
    verifier = replace(verifier, preparations=_DerivedIdWriter())
    inputs_a = tuple(canonical_json_bytes(value) for value in (proposer, document, receipt))
    binding_a = parse_canonical_json(verifier.verify_exact_candidate(*inputs_a).binding)

    payload = document["payload"]
    payload["complete_semantic_head_set"][0]["digest"] = "3" * 64
    payload["finalization_request_id"] = "request-B"
    document["document_digest"] = hashlib.sha256(
        ipc.DOCUMENT_DIGEST_DOMAIN + canonical_json_bytes(payload)
    ).hexdigest()
    proposer["signed_or_authenticated_document_digest"] = document["document_digest"]
    receipt.update(
        accepted_document_digest=document["document_digest"],
        complete_semantic_head_digest=complete_semantic_head_digest(
            payload["complete_semantic_head_set"]
        ),
        finalization_request_id="request-B",
        receipt_id="receipt-B",
    )
    for value, key, domain in (
        (proposer, pkey, ipc.PROPOSER_DOMAIN),
        (document, fkey, ipc.DOCUMENT_AUTHENTICATION_DOMAIN),
        (receipt, fkey, RECEIPT_AUTHENTICATION_DOMAIN),
    ):
        unsigned = {
            name: item for name, item in value.items() if name != "authentication_tag_or_signature"
        }
        value["authentication_tag_or_signature"] = (
            base64.urlsafe_b64encode(key.sign(domain + canonical_json_bytes(unsigned)))
            .rstrip(b"=")
            .decode()
        )
    inputs_b = tuple(canonical_json_bytes(value) for value in (proposer, document, receipt))
    binding_b = parse_canonical_json(verifier.verify_exact_candidate(*inputs_b).binding)
    assert binding_a["proposed_document_digest"] != binding_b["proposed_document_digest"]
    assert binding_a["preparation_id"] != binding_b["preparation_id"]
    return (inputs_a, binding_a), (inputs_b, binding_b)


@pytest.mark.parametrize("response_candidate", [0, 1], ids=["A-for-A", "B-for-A"])
@LINUX
def test_real_prepared_response_must_match_sent_candidate_without_retry(
    tmp_path, monkeypatch, kernel_pidfd, response_candidate
):
    candidates = _two_verified_candidates()
    inputs_a, _binding_a = candidates[0]
    _inputs, response_binding = candidates[response_candidate]
    raw = _prepared_response(response_binding, rehash=False)
    # Obie odpowiedzi mają poprawny własny hash i istniejący schemat.
    assert ipc._decode_response(raw).outcome is ipc.FreshnessIPCOutcome.PREPARED
    with _scripted(tmp_path, _frame(raw)) as server:
        _model_installed_coordinates(monkeypatch, server)
        result = ipc.RuntimeFreshnessVerifierClient(server.expected).verify_exact_candidate(
            *inputs_a
        )
        if response_candidate == 0:
            assert result.outcome is ipc.FreshnessIPCOutcome.PREPARED
            assert result.binding == canonical_json_bytes(response_binding)
        else:
            assert result.outcome is ipc.FreshnessIPCOutcome.OUTCOME_UNKNOWN
            assert result.preparation_id is None and result.binding is None
        request = parse_canonical_json(_finish_count(server))
        assert (
            tuple(
                base64.urlsafe_b64decode(request[name] + "=" * (-len(request[name]) % 4))
                for name in (
                    "proposer_authentication",
                    "authoritative_document",
                    "finalization_receipt",
                )
            )
            == inputs_a
        )


def test_model_both_verified_candidates_match_their_own_binding():
    for inputs, binding in _two_verified_candidates():
        ipc._check_candidate_binding(canonical_json_bytes(binding), inputs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("proposed_document_digest", "f" * 64),
        ("proposer_authentication_digest", "f" * 64),
        ("authoritative_document_authentication_digest", "f" * 64),
        ("receipt_canonical_digest", "f" * 64),
        ("original_decision_identity", "f" * 64),
        ("receipt_id", "another-receipt"),
        ("finalization_request_id", "another-request"),
        ("expected_predecessor_generation", 7),
        ("expected_predecessor_document_digest", "f" * 64),
        ("proposer_identity", "another-proposer"),
        ("proposer_key_version", 7),
        ("finalization_key_version", 7),
        ("trust_domain", "another-domain"),
        ("authority_id", "another-authority"),
    ],
)
def test_model_self_hashed_binding_cannot_change_request_derived_fields(field, value):
    inputs, binding = _two_verified_candidates()[0]
    binding[field] = value
    result = ipc._decode_response(_prepared_response(binding))
    assert result.outcome is ipc.FreshnessIPCOutcome.PREPARED
    with pytest.raises(ValueError, match="PREPARATION_FOR_ANOTHER_CANDIDATE"):
        ipc._check_candidate_binding(result.binding, inputs)
