"""Regresje propozycji: rzeczywisty AF_UNIX, syntetyczne podpisy i model writera.

Nie kwalifikują instalacji, K4 ani transakcji PG. Wariant baseline ma wykazać
błędy przed poprawką; domyślnie badana jest wyłącznie tymczasowa kopia z diffem.
"""

from __future__ import annotations

import multiprocessing
import os
import socket
import struct
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from bot_core import freshness_runtime_ipc as ipc
from bot_core.linux_service_ipc import capture_process_identity
from bot_core.postgresql_freshness_authority import canonical_json_bytes, parse_canonical_json
from tests.security.freshness_verifier_remediation_support import load_verifier
from tests.security.test_freshness_runtime_ipc import (
    _model_installed_coordinates,
    _record_expected,
    _stop_semantic_fixture_process,
)
from tests.security.test_freshness_semantic_verifier import _ipc_request, fixture_candidate

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Wymagane Linux AF_UNIX i fork")

_READ_BUDGET = 0.2


@pytest.fixture
def verifier_module(tmp_path):
    return load_verifier(tmp_path)


def _run_server(
    module,
    address,
    verifier,
    channel,
    calls,
    retained,
    send_errors,
    entered,
    release,
    requests,
    failure,
    budget,
):
    """Instrumentacja pozostaje w osobnym procesie; nie zmienia usług hosta."""
    original_socket = socket.socket

    class Listener(original_socket):
        def listen(self, backlog):
            super().listen(backlog)
            channel.send(("ready", capture_process_identity(os.getpid())))

        def sendall(self, data, flags=0):
            try:
                return super().sendall(data, flags)
            except OSError:
                with send_errors.get_lock():
                    send_errors.value += 1
                raise

    class Writer:
        def prepare(self, binding, _document, _receipt):
            with calls.get_lock():
                calls.value += 1
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Kontroler testu nie zwolnił modelu writera")
            if failure == "unavailable":
                raise OSError("niedostępny model PG")
            with retained.get_lock():
                retained.value += 1
            if failure == "unknown":
                raise module.PreparationOutcomeUnknown("nieznany wynik COMMIT w modelu")
            return parse_canonical_json(binding)["preparation_id"]

    # Model resolvera musi zwrócić dokładny typ badanej kopii modułu.
    verifier.credentials.values = {
        selector: module.RetainedVerificationCredential(**asdict(credential))
        for selector, credential in verifier.credentials.values.items()
    }
    candidate = module.FreshnessSemanticVerifier(
        verifier.credentials, Writer(), verifier.verifier_authority_identity
    )
    module.EXCHANGE_TIMEOUT_SECONDS = budget
    module.socket.socket = Listener
    try:
        module.serve_local_unix_socket(
            candidate, address, allowed_peer_uid=os.getuid(), stop_after=requests
        )
    except Exception as exc:
        channel.send(("error", type(exc).__name__))
    else:
        channel.send(("done", None))
    finally:
        module.socket.socket = original_socket
        channel.close()


@contextmanager
def _server(tmp_path, module, *, requests=1, pause=False, failure=None, budget=_READ_BUDGET):
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe()
    calls = context.Value("i", 0)
    retained = context.Value("i", 0)
    send_errors = context.Value("i", 0)
    entered, release = context.Event(), context.Event()
    if not pause:
        release.set()
    values = fixture_candidate()
    inputs = tuple(canonical_json_bytes(value) for value in values[2:5])
    address = str(tmp_path / "review.sock")
    process = context.Process(
        target=_run_server,
        args=(
            module,
            address,
            values[0],
            child,
            calls,
            retained,
            send_errors,
            entered,
            release,
            requests,
            failure,
            budget,
        ),
    )
    process.start()
    child.close()
    try:
        assert parent.poll(5), "Brak gotowości prywatnego serwera testowego"
        status, record = parent.recv()
        assert status == "ready", record
        yield SimpleNamespace(
            address=address,
            record=record,
            expected=_record_expected(record),
            inputs=inputs,
            frame=struct.pack("!I", len(_ipc_request(values))) + _ipc_request(values),
            calls=calls,
            retained=retained,
            send_errors=send_errors,
            entered=entered,
            release=release,
            channel=parent,
            process=process,
        )
    finally:
        release.set()
        _stop_semantic_fixture_process(process)
        parent.close()


def _connect(server):
    peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    peer.settimeout(1.5)
    peer.connect(server.address)
    return peer


def _exact(peer, length):
    result = bytearray()
    while len(result) < length:
        part = peer.recv(length - len(result))
        assert part, "Niepełna odpowiedź testowanego serwera"
        result.extend(part)
    return bytes(result)


def _response(peer):
    size = struct.unpack("!I", _exact(peer, 4))[0]
    return parse_canonical_json(_exact(peer, size))


def _closed(peer):
    try:
        assert peer.recv(1) == b"", "Po deadline nie wolno wysłać PREPARED ani INVALID"
    except ConnectionResetError:
        pass


def _finished(server):
    assert server.channel.poll(2), "Serwer nie zakończył obsługi limitowanej liczby żądań"
    assert server.channel.recv() == ("done", None), "Błąd połączenia zakończył usługę"
    server.process.join(2)
    assert server.process.exitcode == 0


@pytest.mark.parametrize("partial", [b"\x00", struct.pack("!I", 20) + b"{"])
def test_partial_frame_deadline_closes_without_preparation(tmp_path, verifier_module, partial):
    """Otwarty niepełny nagłówek/body nie może blokować następnego klienta."""
    with _server(tmp_path, verifier_module, requests=2) as server:
        with _connect(server) as slow:
            slow.sendall(partial)
            _closed(slow)
        assert server.calls.value == 0
        with _connect(server) as next_peer:
            next_peer.sendall(struct.pack("!I", 2) + b"{}")
            next_peer.shutdown(socket.SHUT_WR)
            assert _response(next_peer) == {"outcome": "INVALID"}
        _finished(server)


def test_slow_client_cannot_reset_absolute_read_deadline(tmp_path, verifier_module):
    """Kolejne bajty przed każdym timeoutem nie odnawiają budżetu całej ramki."""
    with _server(tmp_path, verifier_module) as server, _connect(server) as slow:
        stop = threading.Event()
        slow.sendall(struct.pack("!I", 1000))

        def drip():
            while not stop.wait(_READ_BUDGET / 4):
                try:
                    slow.sendall(b"x")
                except OSError:
                    return

        sender = threading.Thread(target=drip)
        sender.start()
        try:
            _closed(slow)
            assert server.calls.value == 0
            _finished(server)
        finally:
            stop.set()
            sender.join(2)
            assert not sender.is_alive()


@pytest.mark.parametrize("partial", [b"\x00", struct.pack("!I", 2) + b"{"])
def test_partial_frame_with_eof_is_invalid(tmp_path, verifier_module, partial):
    with _server(tmp_path, verifier_module) as server, _connect(server) as peer:
        peer.sendall(partial)
        peer.shutdown(socket.SHUT_WR)
        assert _response(peer) == {"outcome": "INVALID"}
        assert server.calls.value == 0
        _finished(server)


@pytest.mark.parametrize("extra", [b"x", struct.pack("!I", 2) + b"{}"])
def test_trailing_bytes_are_invalid_before_prepare(tmp_path, verifier_module, extra):
    with _server(tmp_path, verifier_module) as server, _connect(server) as peer:
        peer.sendall(server.frame + extra)
        peer.shutdown(socket.SHUT_WR)
        assert _response(peer) == {"outcome": "INVALID"}
        assert server.calls.value == 0
        _finished(server)


def test_delayed_trailing_byte_is_rejected_before_prepare(tmp_path, verifier_module):
    with _server(tmp_path, verifier_module) as server, _connect(server) as peer:
        peer.sendall(server.frame)
        time.sleep(_READ_BUDGET / 4)
        peer.sendall(b"x")
        peer.shutdown(socket.SHUT_WR)
        assert _response(peer) == {"outcome": "INVALID"}
        assert server.calls.value == 0
        _finished(server)


def test_complete_frame_without_eof_cannot_prepare(tmp_path, verifier_module):
    with _server(tmp_path, verifier_module) as server, _connect(server) as peer:
        peer.sendall(server.frame)
        _closed(peer)
        assert server.calls.value == 0
        _finished(server)


@pytest.mark.parametrize(
    "failure,expected",
    [(None, "PREPARED"), ("unavailable", "UNAVAILABLE"), ("unknown", "OUTCOME_UNKNOWN")],
)
def test_closed_request_preserves_outcomes_without_retry(
    tmp_path, verifier_module, failure, expected
):
    with _server(tmp_path, verifier_module, failure=failure) as server, _connect(server) as peer:
        peer.sendall(server.frame)
        peer.shutdown(socket.SHUT_WR)
        assert _response(peer)["outcome"] == expected
        assert server.calls.value == 1, "Wynik niejednoznaczny nie może uruchamiać retry"
        _finished(server)


def test_disconnect_during_response_does_not_stop_server(tmp_path, verifier_module):
    """Odbiorca znika po przygotowaniu; kolejne połączenie ma być obsłużone."""
    with _server(tmp_path, verifier_module, requests=2, pause=True, budget=1.0) as server:
        with _connect(server) as vanished:
            vanished.sendall(server.frame)
            vanished.shutdown(socket.SHUT_WR)
            assert server.entered.wait(2), "Model writera nie został wywołany"
        server.release.set()
        with _connect(server) as next_peer:
            next_peer.sendall(struct.pack("!I", 2) + b"{}")
            next_peer.shutdown(socket.SHUT_WR)
            assert _response(next_peer) == {"outcome": "INVALID"}
        assert server.calls.value == 1, "Rozłączenie nie cofa przygotowania ani nie uruchamia retry"
        _finished(server)
        assert server.send_errors.value == 1, "Kontrola musi dojść do błędu prawdziwego sendall"
        assert server.retained.value == 1, "Utrata odpowiedzi nie cofa modelowanego efektu"


def test_existing_runtime_client_half_close_is_compatible(tmp_path, verifier_module, monkeypatch):
    """Rzeczywisty klient/AF_UNIX; instalacja i konta są jawnym modelem testu."""
    with _server(tmp_path, verifier_module, requests=2) as server:
        _model_installed_coordinates(monkeypatch, server)
        result = ipc.RuntimeFreshnessVerifierClient(server.expected).verify_exact_candidate(
            *server.inputs
        )
        assert result.outcome is ipc.FreshnessIPCOutcome.PREPARED
        assert server.calls.value == 1
        with _connect(server) as final_peer:
            final_peer.sendall(struct.pack("!I", 2) + b"{}")
            final_peer.shutdown(socket.SHUT_WR)
            assert _response(final_peer) == {"outcome": "INVALID"}
        _finished(server)


def test_client_timeout_does_not_cancel_preparation_or_retry(
    tmp_path, verifier_module, monkeypatch
):
    """Model opóźnionego PG: klient zwraca UNKNOWN, writer nadal jest w toku."""
    with _server(tmp_path, verifier_module, pause=True) as server:
        _model_installed_coordinates(monkeypatch, server)
        monkeypatch.setattr(ipc, "EXCHANGE_TIMEOUT_SECONDS", _READ_BUDGET / 2)
        result = ipc.RuntimeFreshnessVerifierClient(server.expected).verify_exact_candidate(
            *server.inputs
        )
        assert result.outcome is ipc.FreshnessIPCOutcome.OUTCOME_UNKNOWN
        assert server.entered.is_set() and server.calls.value == 1
        assert server.retained.value == 0
        assert server.process.is_alive(), "Timeout klienta nie jest anulowaniem PG"
        time.sleep(_READ_BUDGET)
        server.release.set()
        _finished(server)
        assert server.calls.value == 1
        assert server.retained.value == 1, "Modelowane przygotowanie kończy się po timeout klienta"
