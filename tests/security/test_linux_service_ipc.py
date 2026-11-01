"""Fakty jądra i jawne modele gałęzi odczytowej; bez kwalifikacji produktu."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import multiprocessing
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace

import pytest

from bot_core import linux_service_ipc as ipc


_LINUX = pytest.mark.skipif(sys.platform != "linux", reason="Jądrowe obserwacje wymagają Linux")
_INSTALLATION = "a" * 64


def _request(**changes):
    result = {
        "version": 1,
        "operation": "READONLY_QUALIFY",
        "installation_id": _INSTALLATION,
        "challenge_id": "b" * 64,
        "nonce": "c" * 64,
    }
    result.update(changes)
    return result


def _listener(address, channel):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(address)
            listener.listen(1)
            listener.settimeout(5)
            channel.send(ipc.capture_process_identity(os.getpid()))
            connection, _ = listener.accept()
            with connection:
                assert channel.poll(5), "Brak polecenia zamknięcia rzeczywistego serwera"
                channel.recv()
    finally:
        channel.close()


@contextmanager
def _real_peer(tmp_path):
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe()
    address = str(tmp_path / "pin.sock")
    process = context.Process(target=_listener, args=(address, child))
    process.start()
    child.close()
    try:
        assert parent.poll(5), "Rzeczywisty serwer nie utworzył socketu"
        identity = parent.recv()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
            peer.settimeout(5)
            peer.connect(address)
            yield peer, identity, process, parent
    finally:
        if process.is_alive():
            parent.send("KONIEC")
        parent.close()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
        assert process.exitcode == 0


@_LINUX
def test_capture_matches_real_kernel_process_program_and_cgroup():
    actual = ipc.capture_process_identity(os.getpid())
    assert set(actual) == {
        "pid",
        "uid",
        "gid",
        "groups",
        "start_time_ticks",
        "exe_sha256",
        "cgroup",
    }
    assert (actual["pid"], actual["uid"], actual["gid"]) == (
        os.getpid(),
        os.getuid(),
        os.getgid(),
    )
    assert actual["groups"] == sorted(os.getgroups())
    assert actual["start_time_ticks"] > 0
    assert actual["exe_sha256"] == hashlib.sha256(Path("/proc/self/exe").read_bytes()).hexdigest()
    assert actual["cgroup"] == Path("/proc/self/cgroup").read_text().strip().removeprefix("0::")


@pytest.mark.parametrize("pid", [None, True, False, 0, -1, "1", 1.0])
@_LINUX
def test_capture_refuses_non_exact_or_nonpositive_pid(pid):
    with pytest.raises(ipc.ProductionSignerIPCError, match="PID_REQUIRED"):
        ipc.capture_process_identity(pid)


def test_non_linux_capture_refuses_before_reading_proc(monkeypatch):
    monkeypatch.setattr(ipc.sys, "platform", "win32")
    with pytest.raises(ipc.ProductionSignerIPCError, match="LINUX_PIDFD"):
        ipc.capture_process_identity(1)


@_LINUX
def test_real_connected_peer_has_socket_bound_pidfd_and_exact_observations(tmp_path):
    with _real_peer(tmp_path) as (peer, expected, _process, _channel):
        with ipc.PinnedReadOnlyPeer(peer, uid=os.getuid(), gid=os.getgid()) as pinned:
            assert pinned.credentials == {key: expected[key] for key in ("pid", "uid", "gid")}
            assert pinned.process == expected
            assert pinned.pidfd is not None
            assert not os.get_inheritable(pinned.pidfd)
            pinned.recheck()
        assert pinned.pidfd is None


@pytest.mark.parametrize("field", ["uid", "gid"])
@_LINUX
def test_kernel_rejects_other_expected_identity_before_process_read(tmp_path, field, monkeypatch):
    with _real_peer(tmp_path) as (peer, _identity, _process, _channel):
        expected = {"uid": os.getuid(), "gid": os.getgid()}
        expected[field] += 1
        monkeypatch.setattr(
            ipc,
            "capture_process_identity",
            lambda *_args, **_kwargs: pytest.fail("Odmowa powinna poprzedzać odczyt /proc"),
        )
        with pytest.raises(ipc.ProductionSignerIPCError, match="KERNEL_CLIENT_DENIED"):
            ipc.PinnedReadOnlyPeer(peer, **expected)


@_LINUX
def test_same_process_socket_pair_does_not_prove_an_independent_service():
    left, right = socket.socketpair()
    with left, right:
        with pytest.raises(ipc.ProductionSignerIPCError, match="KERNEL_CLIENT_DENIED"):
            ipc.PinnedReadOnlyPeer(left, uid=os.getuid(), gid=os.getgid())


@_LINUX
def test_non_stream_unix_socket_is_refused_before_credentials():
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as peer:
        with pytest.raises(ipc.ProductionSignerIPCError, match="AF_UNIX_STREAM"):
            ipc.PinnedReadOnlyPeer(peer, uid=os.getuid(), gid=os.getgid())


@_LINUX
def test_dead_origin_is_refused_even_while_connected_socket_is_open(tmp_path):
    with _real_peer(tmp_path) as (peer, _identity, process, channel):
        with ipc.PinnedReadOnlyPeer(peer, uid=os.getuid(), gid=os.getgid()) as pinned:
            channel.send("KONIEC")
            process.join(5)
            assert process.exitcode == 0
            with pytest.raises(ipc.ProductionSignerIPCError, match="PEER_EXITED"):
                pinned.recheck()


@pytest.mark.parametrize("field", ["pid", "uid", "gid", "start_time_ticks", "exe_sha256", "cgroup"])
@_LINUX
def test_pinned_real_peer_refuses_modelled_pid_reuse_program_or_cgroup_change(
    tmp_path, monkeypatch, field
):
    with _real_peer(tmp_path) as (peer, _identity, _process, _channel):
        with ipc.PinnedReadOnlyPeer(peer, uid=os.getuid(), gid=os.getgid()) as pinned:
            altered = copy.deepcopy(pinned.process)
            if field in {"pid", "uid", "gid", "start_time_ticks"}:
                altered[field] += 1
            elif field == "exe_sha256":
                altered[field] = "0" * 64
            else:
                altered[field] += "0::/cgroup-podmieniony\n"
            monkeypatch.setattr(ipc, "capture_process_identity", lambda *_args, **_kwargs: altered)
            with pytest.raises(ipc.ProductionSignerIPCError, match="PEER_CHANGED"):
                pinned.recheck()


@_LINUX
def test_missing_socket_bound_pidfd_never_falls_back_to_pidfd_open(tmp_path, monkeypatch):
    with _real_peer(tmp_path) as (peer, _identity, _process, _channel):
        original = socket.socket.getsockopt

        def no_pidfd(connection, level, option, *args):
            if option == ipc._SO_PEERPIDFD:
                raise OSError("Jądro nie obsługuje SO_PEERPIDFD")
            return original(connection, level, option, *args)

        monkeypatch.setattr(socket.socket, "getsockopt", no_pidfd)
        monkeypatch.setattr(
            os,
            "pidfd_open",
            lambda *_args: pytest.fail("Sam PID nie może zastąpić socket-bound pidfd"),
        )
        with pytest.raises(OSError, match="SO_PEERPIDFD"):
            ipc.PinnedReadOnlyPeer(peer, uid=os.getuid(), gid=os.getgid())


def _serve_qualifier(address, channel, count):
    qualifier = ipc.ReadOnlyVerifierQualifier(_INSTALLATION)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(address)
            listener.listen(4)
            listener.settimeout(5)
            channel.send("GOTOWY")
            for _ in range(count):
                connection, _ = listener.accept()
                with connection:
                    try:
                        qualifier.serve_connection(connection)
                    except (ipc.ProductionSignerIPCError, OSError) as exc:
                        channel.send(("ODMOWA_SERWERA", type(exc).__name__, str(exc)))
                    else:
                        channel.send(("ODPOWIEDZ_APLIKACJI",))
    finally:
        channel.close()


def _call_qualifier(address, value):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
        peer.settimeout(3)
        peer.connect(address)
        ipc._send_frame(peer, value, time.monotonic() + 3)
        peer.shutdown(socket.SHUT_WR)
        return ipc._receive_frame(peer, time.monotonic() + 3)


@_LINUX
@pytest.mark.skipif(
    getattr(os, "geteuid", lambda: 0)() == 0, reason="Test rzeczywistej odmowy nie-root"
)
def test_actual_nonroot_identity_is_rejected_by_server_kernel_policy(tmp_path):
    """Pipe potwierdza mechanizm odmowy serwera, nie tylko błąd po stronie klienta."""
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe()
    address = str(tmp_path / "deny.sock")
    process = context.Process(target=_serve_qualifier, args=(address, child, 1))
    process.start()
    child.close()
    try:
        assert parent.poll(5)
        assert parent.recv() == "GOTOWY"
        with pytest.raises((ipc.ProductionSignerIPCError, OSError)):
            _call_qualifier(address, _request())
        assert parent.poll(5)
        denial = parent.recv()
        assert denial[:2] == ("ODMOWA_SERWERA", "ProductionSignerIPCError")
        assert "QUALIFIER_KERNEL_CLIENT_DENIED" in denial[2]
    finally:
        parent.close()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
        assert process.exitcode == 0


@_LINUX
@pytest.mark.skipif(
    getattr(os, "geteuid", lambda: 1)() != 0
    or os.environ.get("CRYPTOHUNTER_DISPOSABLE_ROOT_IPC") != "1",
    reason="Wymaga jawnego CRYPTOHUNTER_DISPOSABLE_ROOT_IPC=1 w izolowanym root harness",
)
def test_disposable_root_real_application_exchange_and_replay_denial(tmp_path):
    """Opt-in przez uruchomienie w jednorazowym root środowisku; bez zmian hosta."""
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe()
    address = str(tmp_path / "root.sock")
    process = context.Process(target=_serve_qualifier, args=(address, child, 2))
    process.start()
    child.close()
    try:
        assert parent.poll(5)
        assert parent.recv() == "GOTOWY"
        response = _call_qualifier(address, _request())
        assert response["status"] == "PARTIAL_VERIFIER_OBSERVATION"
        assert response["production_qualification"] == "BLOCKED"
        assert response["server_process"]["pid"] == process.pid
        assert response["server_observed_client"]["peer_credentials"] == {
            "pid": os.getpid(),
            "uid": 0,
            "gid": 0,
        }
        assert response["socket_bound_pidfd"] is True
        assert set(ipc.PARTIAL_BLOCKERS) <= set(response["blockers"])
        assert response["postgresql"] is None
        assert parent.poll(5)
        assert parent.recv() == ("ODPOWIEDZ_APLIKACJI",)
        with pytest.raises((ipc.ProductionSignerIPCError, OSError)):
            _call_qualifier(address, _request())
        assert parent.poll(5)
        denial = parent.recv()
        assert denial[0] == "ODMOWA_SERWERA"
        assert "QUALIFIER_REQUEST_REPLAYED" in denial[2]
    finally:
        parent.close()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
        assert process.exitcode == 0


@pytest.fixture
def modeled_qualifier(monkeypatch):
    """Model uwierzytelniania; ten fixture nie jest dowodem poprawnego konta root."""
    identity = {
        "pid": 123,
        "uid": 0,
        "gid": 0,
        "groups": [],
        "start_time_ticks": 10,
        "exe_sha256": "e" * 64,
        "cgroup": "0::/model\n",
    }

    @contextmanager
    def modeled_pin(*_args, **_kwargs):
        yield SimpleNamespace(
            credentials={key: identity[key] for key in ("pid", "uid", "gid")},
            process=identity,
            recheck=lambda: None,
        )

    monkeypatch.setattr(ipc, "PinnedReadOnlyPeer", modeled_pin)
    monkeypatch.setattr(ipc, "capture_process_identity", lambda *_args, **_kwargs: identity)
    return ipc.ReadOnlyVerifierQualifier(_INSTALLATION)


def _modeled_exchange(qualifier, value):
    left, right = socket.socketpair()
    with left, right:
        ipc._send_frame(left, value, time.monotonic() + 1)
        left.shutdown(socket.SHUT_WR)
        qualifier.serve_connection(right)
        right.shutdown(socket.SHUT_WR)
        return ipc._receive_frame(left, time.monotonic() + 1)


@pytest.mark.parametrize(
    "changes",
    [
        {"version": True},
        {"version": 2},
        {"operation": "SIGN"},
        {"operation": "ADMIN"},
        {"operation": "PREPARE"},
        {"installation_id": "0" * 64},
        {"challenge_id": "B" * 64},
        {"nonce": None},
        {"sql": "SELECT 1"},
        {"proposer_authentication": "not-an-authority"},
    ],
)
def test_model_protocol_rejects_sign_admin_sql_and_unbound_requests(modeled_qualifier, changes):
    with pytest.raises(ipc.ProductionSignerIPCError, match="EXACT_READONLY"):
        _modeled_exchange(modeled_qualifier, _request(**changes))


def test_model_success_stays_partial_and_repeated_exact_exchange_refuses(modeled_qualifier):
    response = _modeled_exchange(modeled_qualifier, _request())
    assert response["status"] == "PARTIAL_VERIFIER_OBSERVATION"
    assert response["production_qualification"] == "BLOCKED"
    assert "DURABLE_CHALLENGE_REPLAY_PROTECTION_UNAVAILABLE" in response["blockers"]
    with pytest.raises(ipc.ProductionSignerIPCError, match="REQUEST_REPLAYED"):
        _modeled_exchange(modeled_qualifier, _request())


def test_model_replay_capacity_fails_closed_instead_of_evicting_history(modeled_qualifier):
    modeled_qualifier._used = {(str(index), "d" * 64) for index in range(4096)}
    with pytest.raises(ipc.ProductionSignerIPCError, match="REPLAY_WINDOW_EXHAUSTED"):
        _modeled_exchange(modeled_qualifier, _request())


def test_model_lost_connection_and_timeout_refuse_before_callback(modeled_qualifier, monkeypatch):
    monkeypatch.setattr(ipc, "EXCHANGE_TIMEOUT_SECONDS", 0.03)
    left, right = socket.socketpair()
    with left, right:
        with pytest.raises(TimeoutError):
            modeled_qualifier.serve_connection(right)
        left.shutdown(socket.SHUT_WR)
        with pytest.raises(ipc.ProductionSignerIPCError, match="DISCONNECTED"):
            modeled_qualifier.serve_connection(right)


@_LINUX
def test_installed_policy_refuses_writable_nonroot_ancestry(tmp_path, monkeypatch):
    policy = tmp_path / "qualification.json"
    policy.write_text('{"schema_version":1,"installation_id":"' + _INSTALLATION + '"}')
    monkeypatch.setattr(ipc, "POLICY_PATH", str(policy))
    with pytest.raises(ipc.ProductionSignerIPCError, match="ANCESTRY_UNPROTECTED"):
        ipc.load_installed_qualification_policy()


def _serve_actual_application(address, channel):
    from bot_core.freshness_semantic_verifier import serve_local_unix_socket

    class NoPreparation:
        def verify_exact_candidate(self, *_args):
            raise AssertionError("Diagnostyka nie może uruchamiać operacji produktu")

    try:
        qualifier = ipc.ReadOnlyVerifierQualifier(_INSTALLATION)
        channel.send(ipc.capture_process_identity(os.getpid()))
        serve_local_unix_socket(
            NoPreparation(),
            address,
            allowed_peer_uid=61001,
            stop_after=2,
            socket_gid=0,
            parent_mode=0o750,
            readonly_qualifier=qualifier,
        )
    finally:
        channel.close()


@_LINUX
@pytest.mark.skipif(
    getattr(os, "geteuid", lambda: 1)() != 0
    or os.environ.get("CRYPTOHUNTER_DISPOSABLE_ROOT_IPC") != "1",
    reason="Wymaga jawnego root harness w jednorazowym środowisku",
)
def test_disposable_root_collector_exchanges_with_actual_application_dispatch_and_restart():
    """Rzeczywisty dispatcher i klient; testowe konto root nie kwalifikuje produktu."""
    import tempfile
    from deployment.linux_physical_ipc import ExpectedPeerProcess, collect_verifier_observation

    context = multiprocessing.get_context("fork")
    generations = []
    with tempfile.TemporaryDirectory(prefix="ch-ipc-", dir="/run") as directory:
        os.chmod(directory, 0o750)
        address = directory + "/verifier.sock"
        for _ in range(2):
            parent, child = context.Pipe()
            process = context.Process(target=_serve_actual_application, args=(address, child))
            process.start()
            child.close()
            try:
                assert parent.poll(5)
                observed = parent.recv()
                deadline = time.monotonic() + 5
                while not Path(address).exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                expected = ExpectedPeerProcess(
                    observed["pid"],
                    observed["uid"],
                    observed["gid"],
                    observed["start_time_ticks"],
                    observed["exe_sha256"],
                )
                result = collect_verifier_observation(
                    address,
                    expected=expected,
                    expected_cgroup=observed["cgroup"],
                    challenge_id="b" * 64,
                    installation_id=_INSTALLATION,
                )
                assert result["response"]["server_process"] == observed
                assert (
                    result["response"]["server_observed_client"]["peer_credentials"]["pid"]
                    == os.getpid()
                )
                assert result["production_qualification"] == "BLOCKED"
                generations.append((observed["pid"], observed["start_time_ticks"]))
                with pytest.raises((ipc.ProductionSignerIPCError, OSError)):
                    _call_qualifier(address, {"malformed": True})
            finally:
                parent.close()
                process.join(5)
                if process.is_alive():
                    process.terminate()
                    process.join(5)
                assert process.exitcode == 0
        assert generations[0] != generations[1]


def test_reviewed_verifier_and_adapter_source_pins_match_changed_implementation():
    from bot_core.freshness_deployment_qualification import REVIEWED_CODE_DIGESTS

    root = Path(__file__).resolve().parents[2] / "bot_core"
    for name in (
        "linux_service_ipc.py",
        "linux_service_postgresql.py",
        "production_signer_ipc.py",
        "freshness_semantic_verifier.py",
        "freshness_verifier_service.py",
    ):
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == REVIEWED_CODE_DIGESTS[name]


@pytest.mark.parametrize("enabled", [False, True])
def test_frozen_service_wires_own_connection_and_readonly_observer(monkeypatch, enabled):
    """Model wyłącznie konstrukcji grafu; nie jest dowodem działającej usługi."""
    from bot_core import freshness_verifier_service as service

    process = {"gid": 61002, "exe_sha256": "e" * 64, "cgroup": "/system.slice/verifier.service"}
    monkeypatch.setattr(service.sys, "platform", "linux")
    monkeypatch.setitem(
        sys.modules,
        "pwd",
        SimpleNamespace(
            getpwnam=lambda name: SimpleNamespace(
                pw_uid=61002 if name == service.VERIFIER_USER else 61001
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules, "grp", SimpleNamespace(getgrnam=lambda _: SimpleNamespace(gr_gid=61002))
    )
    monkeypatch.setattr(service.os, "geteuid", lambda: 61002, raising=False)
    monkeypatch.setattr(
        service, "load_installed_qualification_policy", lambda: _INSTALLATION if enabled else None
    )
    monkeypatch.setattr(service, "capture_process_identity", lambda _: process)
    calls = []

    class Database:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def rollback(self):
            calls.append("rollback")

    database = Database()
    factory = SimpleNamespace(connect=lambda: database)
    monkeypatch.setattr(service, "PostgreSQLVerifierConnection", lambda *_args: factory)
    monkeypatch.setattr(service, "FreshnessSemanticVerifier", lambda *_args: object())
    monkeypatch.setattr(service, "PostgreSQLRetainedCredentialResolver", lambda arg: arg)
    monkeypatch.setattr(service, "PostgreSQLPreparationWriter", lambda arg: arg)
    monkeypatch.setattr(
        service,
        "ReadOnlyVerifierQualifier",
        lambda installation, observer: (installation, observer),
    )
    monkeypatch.setattr(
        service,
        "observe_service_connection",
        lambda connection, **kwargs: calls.append((connection, kwargs)) or {"partial": True},
    )

    def serve(_verifier, _path, **kwargs):
        assert kwargs["allowed_peer_uid"] == 61001
        qualifier = kwargs["readonly_qualifier"]
        if enabled:
            assert qualifier[0] == _INSTALLATION
            assert qualifier[1]("b" * 64, _INSTALLATION) == {"partial": True}
        else:
            assert qualifier is None

    monkeypatch.setattr(service, "serve_local_unix_socket", serve)
    service.main()
    if enabled:
        assert calls[0] == "rollback"
        assert calls[1][0] is database
        assert calls[1][1]["expected_database_role"] == "freshness_crypto_verifier"
        assert calls[1][1]["expected_os_principal"] == service.VERIFIER_USER
