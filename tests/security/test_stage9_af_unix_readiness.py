"""Deterministyczne regresje gotowości testowego AF_UNIX; bez usług hosta."""

from __future__ import annotations

import errno
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import multiprocessing
import os
from pathlib import Path
import socket
import stat
import struct
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.security import test_freshness_semantic_verifier as fixture

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Wymagane Linux AF_UNIX i fork")


def _gated_target(verifier, path, requests, ready, *, original, reached, release, listening, die):
    """Zatrzymaj prawdziwy serwer po chmod, ale przed wywołaniem listen."""
    native_socket = socket.socket

    class GatedSocket(native_socket):
        def listen(self, backlog):
            assert Path(path).exists()
            assert stat.S_IMODE(Path(path).stat().st_mode) == 0o660
            reached.set()
            if die:
                os._exit(23)
            assert release.poll(10), "Kontroler nie zwolnił bariery przed listen"
            assert release.recv_bytes() == b"RELEASE"
            super().listen(backlog)
            listening.set()

    # Zmiana jest wyłącznie w osobnym procesie testu, po fork.
    socket.socket = GatedSocket
    try:
        original(verifier, path, requests, ready)
    finally:
        socket.socket = native_socket


def _no_listener_target(_verifier, _path, _requests, ready, *, release):
    """Żywy proces bez bind/listen i bez potwierdzenia gotowości."""
    try:
        assert release.poll(10), "Kontroler nie zakończył testowego procesu"
        assert release.recv_bytes() == b"RELEASE"
    finally:
        ready.close()


def _controlled_target(context, *, die=False):
    reached, listening = context.Event(), context.Event()
    receiver, release = context.Pipe(duplex=False)
    target = partial(
        _gated_target,
        original=fixture._serve_ipc_fixture,
        reached=reached,
        release=receiver,
        listening=listening,
        die=die,
    )
    return target, reached, release, listening


def _assert_no_fixture_children(previous):
    assert {child.pid for child in multiprocessing.active_children()} == previous


def test_reap_keeps_deadline_when_ready_sentinel_does_not_mean_exited(monkeypatch):
    """Nawet gotowy sentinel nie uzasadnia blokującego waitpid."""
    joins, backoffs = [], []
    clock = iter((0.0, 0.1, 0.2))

    class StillAlive:
        exitcode = None

        def join(self, timeout):
            assert timeout == 0, "Dodatni join może zignorować deadline po gotowym sentinelu"
            joins.append(timeout)

    monkeypatch.setattr(
        fixture, "time", SimpleNamespace(monotonic=lambda: next(clock), sleep=backoffs.append)
    )
    process = StillAlive()
    fixture._reap_ipc_process(process, deadline=0.15)
    assert process.exitcode is None and joins
    assert all(0 < interval <= 0.01 for interval in backoffs)


def test_reap_retains_real_exitcode_without_a_blocking_join(monkeypatch):
    joins = []

    class Exiting:
        exitcode = None

        def join(self, timeout):
            assert timeout == 0
            joins.append(timeout)
            if len(joins) == 2:
                self.exitcode = 23

    monkeypatch.setattr(fixture, "time", SimpleNamespace(monotonic=lambda: 0, sleep=lambda _: None))
    process = Exiting()
    fixture._reap_ipc_process(process, deadline=1)
    assert process.exitcode == 23


def test_bound_and_chmodded_socket_refuses_before_listen_with_live_child(tmp_path):
    """Reprodukcja: istniejący socket 0660 + żywy proces nadal daje ECONNREFUSED."""
    context = multiprocessing.get_context("fork")
    target, reached, release, listening = _controlled_target(context)
    receiver, sender = context.Pipe(duplex=False)
    path = tmp_path / "verifier.sock"
    process = context.Process(target=target, args=(fixture.fixture_candidate()[0], path, 1, sender))
    process.start()
    sender.close()
    try:
        assert reached.wait(5), "Nie osiągnięto bariery po chmod"
        assert process.is_alive() and process.exitcode is None
        assert path.is_socket() and stat.S_IMODE(path.stat().st_mode) == 0o660
        assert not listening.is_set() and not receiver.poll()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(1)
            with pytest.raises(ConnectionRefusedError) as refusal:
                client.connect(str(path))
        assert refusal.value.errno == errno.ECONNREFUSED
        assert process.is_alive(), "Odmowa nie wynika ze śmierci dziecka"
        release.send_bytes(b"RELEASE")
        assert receiver.poll(5), "Nie nadeszło potwierdzenie po listen"
        assert receiver.recv_bytes() == b"LISTENING" and listening.is_set()
        assert fixture._ipc_exchange(path, b"{}") == {"outcome": "INVALID"}
        fixture._reap_ipc_process(process, deadline=time.monotonic() + 5.0)
        assert process.exitcode == 0 and not path.exists()
    finally:
        release.send_bytes(b"RELEASE")
        if process.is_alive():
            process.terminate()
        fixture._reap_ipc_process(process, deadline=time.monotonic() + 5.0)
        assert not process.is_alive()
        receiver.close()
        if path.is_socket():
            path.unlink()


def test_helper_waits_for_delayed_listen_without_a_connect_probe(tmp_path, monkeypatch):
    context = multiprocessing.get_context("fork")
    target, reached, release, listening = _controlled_target(context)
    monkeypatch.setattr(fixture, "_serve_ipc_fixture", target)
    values = list(fixture.fixture_candidate())

    def controller():
        try:
            assert reached.wait(5), "Dziecko nie osiągnęło bariery"
            assert not listening.is_set()
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                with pytest.raises(ConnectionRefusedError):
                    client.connect(str(tmp_path / "verifier.sock"))
        finally:
            release.send_bytes(b"RELEASE")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(controller)
        try:
            with fixture._ipc_server(tmp_path, values[0]) as (path, process):
                future.result(timeout=5)
                assert listening.is_set() and process.is_alive()
                response = fixture._ipc_exchange(path, fixture._ipc_request(values))
                assert response["outcome"] == "PREPARED"
                assert response["preparation_id"] == "opaque-preparation"
                assert set(response) == {"outcome", "preparation_id", "binding"}
        finally:
            release.send_bytes(b"RELEASE")


def test_helper_reports_child_exit_before_listen_and_removes_stale_socket(tmp_path, monkeypatch):
    context = multiprocessing.get_context("fork")
    target, reached, _release, listening = _controlled_target(context, die=True)
    monkeypatch.setattr(fixture, "_serve_ipc_fixture", target)
    previous = {child.pid for child in multiprocessing.active_children()}
    with pytest.raises(pytest.fail.Exception) as failure:
        with fixture._ipc_server(tmp_path, fixture.fixture_candidate()[0], readiness_timeout=2):
            pytest.fail("Helper przyjął gotowość bez listen")
    assert reached.is_set() and not listening.is_set()
    assert "exitcode=23" in str(failure.value)
    assert "socket_exists=True" in str(failure.value)
    assert not (tmp_path / "verifier.sock").exists()
    _assert_no_fixture_children(previous)


@pytest.mark.parametrize("bound", [False, True], ids=["no-socket", "bound-without-listen"])
def test_helper_has_bounded_failure_when_listener_never_starts(tmp_path, monkeypatch, bound):
    context = multiprocessing.get_context("fork")
    previous = {child.pid for child in multiprocessing.active_children()}
    if bound:
        target, reached, release, listening = _controlled_target(context)
    else:
        reached, listening = context.Event(), context.Event()
        receiver, release = context.Pipe(duplex=False)
        target = partial(_no_listener_target, release=receiver)
    monkeypatch.setattr(fixture, "_serve_ipc_fixture", target)
    try:
        with pytest.raises(pytest.fail.Exception, match="limit oczekiwania") as failure:
            with fixture._ipc_server(tmp_path, fixture.fixture_candidate()[0], readiness_timeout=2):
                pytest.fail("Helper przyjął gotowość bez listenera")
        assert "alive=True" in str(failure.value)
        assert f"socket_exists={bound}" in str(failure.value)
        assert not listening.is_set()
        if bound:
            assert reached.is_set()
        assert not (tmp_path / "verifier.sock").exists()
        _assert_no_fixture_children(previous)
    finally:
        release.send_bytes(b"RELEASE")


def _lost_response_target(verifier, path, requests, ready, *, original, accepts, preparations):
    """Rzeczywiste przygotowanie fixture, potem wymuszone EOF zamiast odpowiedzi."""
    writer = verifier.preparations

    class CountedWriter:
        def prepare(self, *arguments):
            with preparations.get_lock():
                preparations.value += 1
            return writer.prepare(*arguments)

    object.__setattr__(verifier, "preparations", CountedWriter())
    native_socket = socket.socket

    class LostResponseSocket(native_socket):
        def accept(self):
            peer = super().accept()
            with accepts.get_lock():
                accepts.value += 1
            return peer

        def sendall(self, data, flags=0):
            # Fault injection: nie wysyłaj odpowiedzi po zakończonym prepare.
            self.shutdown(socket.SHUT_WR)

    socket.socket = LostResponseSocket
    try:
        original(verifier, path, requests, ready)
    finally:
        socket.socket = native_socket


def test_response_loss_after_send_never_reconnects_or_prepares_twice(tmp_path, monkeypatch):
    context = multiprocessing.get_context("fork")
    accepts, preparations = context.Value("i", 0), context.Value("i", 0)
    target = partial(
        _lost_response_target,
        original=fixture._serve_ipc_fixture,
        accepts=accepts,
        preparations=preparations,
    )
    monkeypatch.setattr(fixture, "_serve_ipc_fixture", target)
    values = list(fixture.fixture_candidate())
    connects, sends = [], []
    native_socket = socket.socket

    class CountedClient(native_socket):
        def connect(self, address):
            connects.append(address)
            return super().connect(address)

        def sendall(self, data, flags=0):
            sends.append(len(data))
            return super().sendall(data, flags)

    with fixture._ipc_server(tmp_path, values[0]) as (path, _process):
        proxy = SimpleNamespace(**{**vars(socket), "socket": CountedClient})
        with patch.object(fixture, "socket", proxy):
            with pytest.raises(EOFError, match="przed końcem odpowiedzi"):
                fixture._ipc_exchange(path, fixture._ipc_request(values))
        assert connects == [str(path)] and len(sends) == 1 and sends[0] > 0
    assert accepts.value == preparations.value == 1


def _fragmented_response_target(
    _verifier, path, _requests, ready, *, response, offsets, attempts, accepts
):
    """TEST_ONLY: rzeczywisty transport, modelowa odpowiedź bez prepare/PG."""
    payload = json.dumps(response).encode()
    frame = struct.pack("!I", len(payload)) + payload
    fragments = (frame[:1], frame[1:3], frame[3:8], frame[8:])

    def receive_exact(peer, count):
        result = bytearray()
        while len(result) < count:
            part = peer.recv(count - len(result))
            assert part, "Klient zakończył niepełną ramkę żądania"
            result.extend(part)
        return bytes(result)

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.settimeout(5)
            listener.bind(str(path))
            listener.listen(1)
            ready.send_bytes(b"LISTENING")
            ready.close()
            peer, _address = listener.accept()
            with peer:
                accepts.value += 1
                peer.settimeout(5)
                size = struct.unpack("!I", receive_exact(peer, 4))[0]
                assert receive_exact(peer, size) == b"{}"
                started = time.monotonic()
                for index, (offset, fragment) in enumerate(zip(offsets, fragments, strict=True)):
                    # Opóźnienia są wektorem transmisji; gotowość potwierdza Pipe po listen.
                    time.sleep(max(0, started + offset - time.monotonic()))
                    attempts[index] = time.monotonic()
                    try:
                        peer.sendall(fragment)
                    except (BrokenPipeError, ConnectionResetError):
                        break
    finally:
        ready.close()
        path.unlink(missing_ok=True)


def test_fragmented_response_cannot_renew_total_exchange_deadline(tmp_path, monkeypatch):
    context = multiprocessing.get_context("fork")
    accepts, attempts = context.Value("i", 0), context.Array("d", 4)
    monkeypatch.setattr(
        fixture,
        "_serve_ipc_fixture",
        partial(
            _fragmented_response_target,
            response={"outcome": "INVALID"},
            offsets=(0, 0.9, 1.8, 3.6),
            attempts=attempts,
            accepts=accepts,
        ),
    )
    connects, sends = [], []

    class CountedClient(socket.socket):
        def connect(self, address):
            connects.append(address)
            return super().connect(address)

        def sendall(self, data, flags=0):
            sends.append(data)
            return super().sendall(data, flags)

    with fixture._ipc_server(tmp_path, fixture.fixture_candidate()[0]) as (path, _process):
        proxy = SimpleNamespace(**{**vars(socket), "socket": CountedClient})
        with patch.object(fixture, "socket", proxy):
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                fixture._ipc_exchange(path, b"{}")
            elapsed = time.monotonic() - started
        assert 2.5 <= elapsed < 4.5
        assert connects == [str(path)] and sends == [struct.pack("!I", 2) + b"{}"]
    assert accepts.value == 1
    timestamps = list(attempts)
    assert all(timestamps)
    assert timestamps[-1] - timestamps[0] > fixture.EXCHANGE_TIMEOUT_SECONDS
    assert max(later - earlier for earlier, later in zip(timestamps, timestamps[1:])) < 3


@pytest.mark.parametrize(
    "response",
    [
        {"outcome": "INVALID"},
        {"outcome": "UNAVAILABLE"},
        {"outcome": "OUTCOME_UNKNOWN"},
        {"outcome": "PREPARED", "preparation_id": "opaque-preparation", "binding": "e30"},
    ],
    ids=["INVALID", "UNAVAILABLE", "OUTCOME_UNKNOWN", "PREPARED"],
)
def test_fragmented_response_within_total_deadline_keeps_exact_result(
    tmp_path, monkeypatch, response
):
    context = multiprocessing.get_context("fork")
    accepts, attempts = context.Value("i", 0), context.Array("d", 4)
    target = partial(
        _fragmented_response_target,
        response=response,
        offsets=(0, 0.02, 0.04, 0.06),
        attempts=attempts,
        accepts=accepts,
    )
    monkeypatch.setattr(fixture, "_serve_ipc_fixture", target)
    with fixture._ipc_server(tmp_path, fixture.fixture_candidate()[0]) as (path, _process):
        assert fixture._ipc_exchange(path, b"{}") == response
    assert accepts.value == 1 and all(attempts)
    assert attempts[-1] - attempts[0] < fixture.EXCHANGE_TIMEOUT_SECONDS


@pytest.mark.parametrize("late_operation", ["connect", "sendall", "last_recv"])
def test_exchange_rejects_operation_returning_after_shared_deadline(monkeypatch, late_operation):
    """Model zegara sprawdza zużycie budżetu także przez zakończoną operację."""
    now, operations, timeouts = [0.0], [], []
    payload = b'{"outcome":"INVALID"}'
    fragments = [struct.pack("!I", len(payload)), payload]

    def operation(name):
        operations.append(name)
        now[0] += 3.1 if name == late_operation else 0.2

    class ModelClient:
        def __enter__(self):
            return self

        def __exit__(self, *_arguments):
            return None

        def settimeout(self, timeout):
            assert 0 < timeout <= fixture.EXCHANGE_TIMEOUT_SECONDS
            timeouts.append(timeout)

        def connect(self, _address):
            operation("connect")

        def sendall(self, _data):
            operation("sendall")

        def recv(self, count):
            part = fragments.pop(0)
            assert count == len(part)
            operation("header_recv" if fragments else "last_recv")
            return part

    proxy = SimpleNamespace(**{**vars(socket), "socket": lambda *_args: ModelClient()})
    monkeypatch.setattr(fixture, "socket", proxy)
    monkeypatch.setattr(fixture, "time", SimpleNamespace(monotonic=lambda: now[0]))
    with pytest.raises(TimeoutError):
        fixture._ipc_exchange(Path("model.sock"), b"{}")
    expected = ["connect", "sendall", "header_recv", "last_recv"]
    assert operations == expected[: expected.index(late_operation) + 1]
    assert len(timeouts) == len(operations)
    assert timeouts == pytest.approx([3 - 0.2 * index for index in range(len(operations))])
