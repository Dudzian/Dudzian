"""Rzeczywiste fakty AF_UNIX nie nadają testom prawa kwalifikacji produktu."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import multiprocessing
import os
from pathlib import Path
import socket
import struct
import sys
from types import SimpleNamespace

import pytest

from deployment import linux_physical_ipc as ipc


@pytest.mark.parametrize(
    "pid,uid,gid", [(0, 1, 2), (-1, 1, 2), (True, 1, 2), (1, -1, 2), (1, 1, False)]
)
def test_kernel_credentials_refuse_invalid_identity(pid, uid, gid):
    with pytest.raises(ipc.LinuxIPCObservationError):
        ipc.KernelPeerCredentials(pid, uid, gid)


@pytest.mark.parametrize(
    "start,digest", [(0, "a" * 64), (True, "a" * 64), (1, "A" * 64), (1, "a" * 63), (1, None)]
)
def test_expected_process_requires_exact_lifetime_and_program(start, digest):
    with pytest.raises(ipc.LinuxIPCObservationError):
        ipc.ExpectedPeerProcess(10, 20, 30, start, digest)


@pytest.mark.parametrize(
    "value", [None, {}, [], True, "PASS", {"status": "PASS"}, {"socket_mode": 0o660, "uid": 1001}]
)
def test_metadata_or_pass_cannot_replace_application_authentication(value):
    with pytest.raises(ipc.LinuxIPCQualificationBlocked):
        ipc.require_ipc_qualification(value)


def test_even_exact_blocked_diagnostic_is_not_qualification():
    result = ipc.blocked_ipc_qualification()
    assert result["status"] == "BLOCKED"
    assert "ACTUAL_RUNTIME_CLIENT_EXCHANGE_UNPROVEN" in result["blockers"]
    with pytest.raises(ipc.LinuxIPCQualificationBlocked, match="funkcjonalnego"):
        ipc.require_ipc_qualification(result)
    result["blockers"].clear()
    assert ipc.blocked_ipc_qualification()["blockers"]


def test_unsupported_platform_refuses_before_socket_access(monkeypatch):
    monkeypatch.setattr(ipc.sys, "platform", "win32")
    with pytest.raises(ipc.LinuxIPCObservationError, match="natywnego Linux"):
        ipc.observe_kernel_peer(None)


@pytest.mark.skipif(sys.platform != "linux", reason="SO_PEERCRED jest mechanizmem Linux")
def test_non_stream_or_non_unix_objects_refuse():
    with pytest.raises(ipc.LinuxIPCObservationError, match="AF_UNIX"):
        ipc.observe_kernel_peer(None)
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as peer:
        with pytest.raises(ipc.LinuxIPCObservationError, match="AF_UNIX"):
            ipc.observe_kernel_peer(peer)


@pytest.mark.skipif(sys.platform != "linux", reason="SO_PEERCRED jest mechanizmem Linux")
@pytest.mark.parametrize("raw", [b"", b"short", struct.pack("3i", -1, 0, 0)])
def test_malformed_kernel_response_refuses(monkeypatch, raw):
    monkeypatch.setattr(socket.socket, "getsockopt", lambda *_args: raw)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
        with pytest.raises(ipc.LinuxIPCObservationError):
            ipc.observe_kernel_peer(peer)


def _server(path, channel):
    """Dwa procesy tego samego konta; żadna fikcyjna separacja UID/GID."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(path)
            listener.listen(1)
            listener.settimeout(3)
            channel.send({"pid": os.getpid(), "uid": os.getuid(), "gid": os.getgid()})
            peer, _ = listener.accept()
            with peer:
                credentials = ipc.observe_kernel_peer(peer)
                channel.send(
                    {"pid": credentials.pid, "uid": credentials.uid, "gid": credentials.gid}
                )
                if not channel.poll(3):
                    raise RuntimeError("Brak zamknięcia testowego połączenia")
                channel.recv()
    finally:
        channel.close()


@contextmanager
def _connection(tmp_path):
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe()
    path = str(tmp_path / "readonly-peer.sock")
    server = context.Process(target=_server, args=(path, child))
    server.start()
    child.close()
    try:
        assert parent.poll(3)
        server_identity = parent.recv()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
            peer.settimeout(3)
            peer.connect(path)
            assert parent.poll(3)
            observed_client = parent.recv()
            yield peer, server_identity, observed_client
    finally:
        if server.is_alive():
            parent.send("KONIEC")
        parent.close()
        server.join(3)
        if server.is_alive():
            server.terminate()
            server.join(3)
        assert server.exitcode == 0


@pytest.mark.skipif(sys.platform != "linux", reason="Rzeczywiste procesy i AF_UNIX na Linux")
def test_real_unprivileged_connection_has_kernel_identity_on_both_ends(tmp_path):
    with _connection(tmp_path) as (peer, server, client):
        actual = ipc.observe_kernel_peer(peer)
        assert (actual.pid, actual.uid, actual.gid) == (server["pid"], server["uid"], server["gid"])
        assert actual.pid != os.getpid()
        assert client == {"pid": os.getpid(), "uid": os.getuid(), "gid": os.getgid()}
        with pytest.raises(ipc.LinuxIPCQualificationBlocked):
            ipc.require_ipc_qualification({"peer_credentials": client, "status": "PASS"})


@pytest.mark.skipif(sys.platform != "linux", reason="Rzeczywiste procesy i AF_UNIX na Linux")
@pytest.mark.parametrize("field", ["pid", "uid", "gid"])
def test_real_connection_with_other_expected_identity_is_denied(tmp_path, field):
    with _connection(tmp_path) as (peer, server, _client):
        expected = dict(server)
        expected[field] += 1
        spec = ipc.ExpectedPeerProcess(**expected, start_time_ticks=1, executable_sha256="a" * 64)
        with pytest.raises(ipc.LinuxIPCObservationError, match="przypiętego procesu"):
            ipc.observe_pinned_peer(peer, spec)


@pytest.mark.skipif(sys.platform != "linux", reason="Rzeczywiste procesy i AF_UNIX na Linux")
def test_same_process_socket_cannot_claim_independent_process():
    left, right = socket.socketpair()
    with left, right:
        spec = ipc.ExpectedPeerProcess(os.getpid(), os.getuid(), os.getgid(), 1, "a" * 64)
        with pytest.raises(ipc.LinuxIPCObservationError, match="odrębnym procesem"):
            ipc.observe_pinned_peer(right, spec)


@pytest.mark.skipif(sys.platform != "linux", reason="Rzeczywiste procesy i AF_UNIX na Linux")
def test_real_peer_pidfd_unavailable_refuses_without_promoting_credentials(tmp_path):
    from deployment.linux_physical_native import process_observation

    with _connection(tmp_path) as (peer, server, _client):
        actual = process_observation(server["pid"])
        spec = ipc.ExpectedPeerProcess(
            **server,
            start_time_ticks=actual["start_time_ticks"],
            executable_sha256=actual["exe_sha256"],
        )
        try:
            result = ipc.observe_pinned_peer(peer, spec)
        except ipc.LinuxIPCObservationError as exc:
            # Brak SO_PEERPIDFD lub dostępu do wymaganych faktów jest odmową,
            # nigdy fallbackiem do pidfd_open PID ani promocją do PASS.
            assert "przypiąć" in str(exc)
        else:
            assert result["status"] == "UNQUALIFIED_TRANSPORT_OBSERVATION"
            assert result["production_qualification"] == "BLOCKED"
            assert result["peer_process"]["exe_sha256"] == actual["exe_sha256"]


@pytest.mark.skipif(sys.platform != "linux", reason="Model pinningu transportu Linux")
@pytest.mark.parametrize(
    "mutation", ["uid", "gid", "pid", "start_time_ticks", "exe_sha256", "second_read", "pidfd_lost"]
)
def test_pinned_transport_model_refuses_process_substitution(tmp_path, monkeypatch, mutation):
    from bot_core import production_signer_ipc
    from deployment import linux_physical_native

    with _connection(tmp_path) as (peer, server, _client):
        first = linux_physical_native.process_observation(server["pid"])
        spec = ipc.ExpectedPeerProcess(
            **server,
            start_time_ticks=first["start_time_ticks"],
            executable_sha256=first["exe_sha256"],
        )
        count = 0

        class PinnedModel:
            def __init__(self, *_args):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def recheck(self):
                if mutation == "pidfd_lost":
                    raise production_signer_ipc.ProductionSignerIPCError("Model utraty pidfd")

        def observation(_pid):
            nonlocal count
            count += 1
            observed = copy.deepcopy(first)
            if mutation == "second_read" and count > 1:
                observed["start_time_ticks"] += 1
            elif mutation in {"uid", "gid", "pid", "start_time_ticks"}:
                observed[mutation] += 1
            elif mutation == "exe_sha256":
                observed[mutation] = "a" * 64
            return observed

        monkeypatch.setattr(production_signer_ipc, "_PinnedPeer", PinnedModel)
        monkeypatch.setattr(linux_physical_native, "process_observation", observation)
        with pytest.raises(ipc.LinuxIPCObservationError):
            ipc.observe_pinned_peer(peer, spec)


@pytest.mark.skipif(sys.platform != "linux", reason="Model pinningu transportu Linux")
def test_even_complete_pinned_transport_model_is_not_functional_qualification(
    tmp_path, monkeypatch
):
    from bot_core import production_signer_ipc
    from deployment import linux_physical_native

    with _connection(tmp_path) as (peer, server, _client):
        actual = linux_physical_native.process_observation(server["pid"])
        spec = ipc.ExpectedPeerProcess(
            **server,
            start_time_ticks=actual["start_time_ticks"],
            executable_sha256=actual["exe_sha256"],
        )

        @contextmanager
        def pinned_model(*_args):
            yield SimpleNamespace(recheck=lambda: None)

        monkeypatch.setattr(production_signer_ipc, "_PinnedPeer", pinned_model)
        result = ipc.observe_pinned_peer(peer, spec)
        assert result["socket_bound_pidfd"] is True
        assert result["production_qualification"] == "BLOCKED"
        with pytest.raises(ipc.LinuxIPCQualificationBlocked):
            ipc.require_ipc_qualification(result)
