"""TEST_ONLY: AccountGenesis i cztery poprawki verifiera przez rzeczywisty AF_UNIX.

Socket, send/recv, half-close, SO_PEERCRED, SO_PEERPIDFD i procfs są rzeczywiste.
NSS, chroniona instalacja, cgroup/grupy usługi oraz PG/CAS są jawnymi modelami.
Kontroler czyta procfs znanego PID utworzonego potomka przed połączeniem:
to model niezależnie zatwierdzonego provisioningu, nigdy odpowiedź verifiera.
Nie jest to natywny dowód dwóch UID; ten nadal pozostaje NOT_RUN.
"""

from __future__ import annotations

import base64
import hashlib
import multiprocessing
import socket
import struct
import sys
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bot_core import (
    account_genesis_freshness_authority as original,
    freshness_runtime_ipc as ipc,
    linux_service_ipc as process_identity,
)
from bot_core.postgresql_freshness_authority import canonical_json_bytes, parse_canonical_json
from tests.security import test_freshness_runtime_ipc as runtime_tests
from tests.security.account_genesis_transport_review_support import load_account_genesis
from tests.security.freshness_verifier_remediation_support import load_verifier
from tests.security.test_freshness_semantic_verifier import fixture_candidate
from tests.security.test_freshness_verifier_transport_remediation import (
    _closed,
    _connect,
    _finished,
    _response,
    _server,
)

kernel_pidfd = runtime_tests.kernel_pidfd
pytestmark = [
    pytest.mark.skipif(sys.platform != "linux", reason="Wymagane Linux AF_UNIX i fork"),
    pytest.mark.usefixtures("kernel_pidfd"),
]


@pytest.fixture
def candidate(tmp_path):
    """Dwa dokładne patche stosowane wyłącznie w prywatnych katalogach testu."""
    return SimpleNamespace(
        account=load_account_genesis(tmp_path / "account"),
        verifier=load_verifier(tmp_path / "verifier"),
    )


def _authority(candidate, server, monkeypatch, *, expected=None):
    module = candidate.account
    configuration = module.ProductionLocalFreshnessAuthorityConfig(
        module.VERIFIER_SOCKET_PATH, "/test/pg", 5432
    )
    # PID pochodzi z utworzonego przez kontroler Process, a nie z odpowiedzi
    # AF_UNIX ani z Pipe. Odczyt procfs jest rzeczywisty; zatwierdzenie źródła
    # przez właściciela oraz tożsamość innego UID pozostają MODELEM.
    if expected is None:
        observation = process_identity.capture_process_identity(server.process.pid)
        expected = runtime_tests._record_expected(observation)
    authority = module.ProductionLocalFreshnessAuthority(
        configuration,
        expected_verifier=expected,
    )
    cas = Mock(
        return_value=module.FreshnessAuthorityResult(module.FreshnessAuthorityOutcome.CAS_ACCEPTED)
    )
    monkeypatch.setattr(authority, "_compare_and_advance", cas)
    return authority, cas


def _finish_listener(server):
    """Utrzymaj proces do końca recheck klienta; druga próba jest tylko kontrolą."""
    with _connect(server) as control:
        control.sendall(struct.pack("!I", 2) + b"{}")
        control.shutdown(socket.SHUT_WR)
        assert _response(control) == {"outcome": "INVALID"}
    _finished(server)


def test_real_adapter_half_close_and_three_objects_reach_separate_cas(
    tmp_path, monkeypatch, candidate
):
    """Pełna kryptografia i transport; przygotowanie oraz CAS są licznikami modeli."""
    with _server(tmp_path, candidate.verifier, requests=2) as server:
        runtime_tests._model_installed_coordinates(monkeypatch, server)
        authority, cas = _authority(candidate, server, monkeypatch)

        result = authority.authenticate_and_advance(*server.inputs)

        assert result.outcome is candidate.account.FreshnessAuthorityOutcome.CAS_ACCEPTED
        assert server.calls.value == server.retained.value == 1
        cas.assert_called_once()
        prepared, document, receipt = cas.call_args.args
        assert prepared.preparation_id == parse_canonical_json(prepared.binding)["preparation_id"]
        assert (document, receipt) == server.inputs[1:]
        ipc._check_candidate_binding(prepared.binding, server.inputs)
        _finish_listener(server)


def test_real_legacy_client_without_eof_is_refused_before_prepare(tmp_path, candidate):
    """Rzeczywisty stary klient bez shutdown, ale tylko jedna próba, bez starego retry."""
    with _server(tmp_path, candidate.verifier) as server:
        authority = original.ProductionLocalFreshnessAuthority(
            original.ProductionLocalFreshnessAuthorityConfig(server.address, "/test/pg", 5432)
        )
        result = authority._request_preparation_once(*server.inputs)
        assert result.outcome is original.FreshnessAuthorityOutcome.PREPARATION_OUTCOME_UNKNOWN
        assert server.calls.value == server.retained.value == 0
        _finished(server)


@pytest.mark.parametrize(
    "failure,expected,retained",
    [
        ("unavailable", "UNAVAILABLE", 0),
        ("unknown", "PREPARATION_OUTCOME_UNKNOWN", 1),
    ],
)
def test_real_adapter_maps_patched_server_failure_without_retry(
    tmp_path, monkeypatch, candidate, failure, expected, retained
):
    with _server(tmp_path, candidate.verifier, requests=2, failure=failure) as server:
        runtime_tests._model_installed_coordinates(monkeypatch, server)
        authority, cas = _authority(candidate, server, monkeypatch)

        result = authority.authenticate_and_advance(*server.inputs)

        assert result.outcome is candidate.account.FreshnessAuthorityOutcome[expected]
        assert server.calls.value == 1
        assert server.retained.value == retained
        cas.assert_not_called()
        _finish_listener(server)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_real_adapter_invalid_signed_input_never_prepares_or_calls_cas(
    tmp_path, monkeypatch, candidate, index
):
    """Trzy podpisane obiekty sprawdza rzeczywisty verifier, bez skutków SQL."""
    with _server(tmp_path, candidate.verifier, requests=2) as server:
        runtime_tests._model_installed_coordinates(monkeypatch, server)
        authority, cas = _authority(candidate, server, monkeypatch)
        inputs = list(server.inputs)
        inputs[index] = b"{}"

        result = authority.authenticate_and_advance(*inputs)

        assert result.outcome is candidate.account.FreshnessAuthorityOutcome.INVALID_AUTHENTICATION
        assert server.calls.value == server.retained.value == 0
        cas.assert_not_called()
        _finish_listener(server)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(b"", id="empty"),
        pytest.param(None, id="none"),
        pytest.param(bytearray(b"{}"), id="mutable"),
        pytest.param(b"x" * (ipc.MAX_FRAME_BYTES + 1), id="too-large"),
    ],
)
def test_real_adapter_local_bad_input_never_connects(tmp_path, monkeypatch, candidate, value):
    """Prawdziwy listener liczy połączenia; dane odrzucane lokalnie przed send."""
    with runtime_tests._scripted(tmp_path) as server:
        runtime_tests._model_installed_coordinates(monkeypatch, server)
        approved = runtime_tests._record_expected(
            process_identity.capture_process_identity(server.process.pid)
        )
        authority, cas = _authority(candidate, server, monkeypatch, expected=approved)

        result = authority.authenticate_and_advance(value, b"{}", b"{}")

        assert result.outcome is candidate.account.FreshnessAuthorityOutcome.INVALID_DOCUMENT
        cas.assert_not_called()
        assert runtime_tests._finish_count(server, connections=0) == b""


def test_real_timeout_keeps_late_effect_and_next_adapter_client_works(
    tmp_path, monkeypatch, candidate
):
    """Timeout socketu nie anuluje modelowanego COMMIT; listener przeżywa błąd sendall."""
    with _server(tmp_path, candidate.verifier, requests=3, pause=True) as server:
        runtime_tests._model_installed_coordinates(monkeypatch, server)
        authority, cas = _authority(candidate, server, monkeypatch)
        monkeypatch.setattr(ipc, "EXCHANGE_TIMEOUT_SECONDS", 0.1)

        result = authority.authenticate_and_advance(*server.inputs)

        assert (
            result.outcome
            is candidate.account.FreshnessAuthorityOutcome.PREPARATION_OUTCOME_UNKNOWN
        )
        assert server.entered.is_set() and server.calls.value == 1
        assert server.retained.value == 0 and server.process.is_alive()
        cas.assert_not_called()
        # Po zamknięciu kanału efekt writera nadal może się utrwalić. Wspólny
        # deadline serwera jest już wyczerpany; błąd odpowiedzi nie kończy usługi.
        time.sleep(0.2)
        server.release.set()
        monkeypatch.setattr(ipc, "EXCHANGE_TIMEOUT_SECONDS", 1.0)

        next_result = authority.authenticate_and_advance(*server.inputs)

        assert next_result.outcome is candidate.account.FreshnessAuthorityOutcome.CAS_ACCEPTED
        assert server.calls.value == server.retained.value == 2
        assert server.send_errors.value == 0  # deadline odmawia jeszcze przed sendall
        cas.assert_called_once()
        _finish_listener(server)


def test_real_response_broken_pipe_after_prepare_keeps_adapter_listener_usable(
    tmp_path, monkeypatch, candidate
):
    """Prawdziwy BrokenPipe po modelu prepare, potem udana operacja adaptera."""
    with _server(tmp_path, candidate.verifier, requests=3, pause=True, budget=1.0) as server:
        with _connect(server) as vanished:
            vanished.sendall(server.frame)
            vanished.shutdown(socket.SHUT_WR)
            assert server.entered.wait(2)
        server.release.set()
        runtime_tests._model_installed_coordinates(monkeypatch, server)
        authority, cas = _authority(candidate, server, monkeypatch)

        result = authority.authenticate_and_advance(*server.inputs)

        assert result.outcome is candidate.account.FreshnessAuthorityOutcome.CAS_ACCEPTED
        assert server.send_errors.value == 1
        assert server.retained.value == server.calls.value == 2
        cas.assert_called_once()
        _finish_listener(server)


@pytest.mark.parametrize("partial", [b"\x00", struct.pack("!I", 20) + b"{"])
def test_real_deadline_frees_listener_for_adapter(tmp_path, monkeypatch, candidate, partial):
    """Niepełna ramka ma rzeczywisty deadline; następnie działa proponowany adapter."""
    with _server(tmp_path, candidate.verifier, requests=3) as server:
        with _connect(server) as slow:
            slow.sendall(partial)
            _closed(slow)
        assert server.calls.value == 0
        runtime_tests._model_installed_coordinates(monkeypatch, server)
        authority, cas = _authority(candidate, server, monkeypatch)

        result = authority.authenticate_and_advance(*server.inputs)

        assert result.outcome is candidate.account.FreshnessAuthorityOutcome.CAS_ACCEPTED
        assert server.calls.value == 1
        cas.assert_called_once()
        _finish_listener(server)


@pytest.mark.parametrize("phase", ["execute", "fetchone"])
def test_real_adapter_select_failure_closes_model_session_before_next_prepare(
    tmp_path, monkeypatch, candidate, phase
):
    """Rzeczywisty AF_UNIX + poprawiony connect, z modelem samej granicy psycopg."""
    context = multiprocessing.get_context("fork")
    closes, connects = context.Value("i", 0), context.Value("i", 0)
    module = candidate.verifier
    original_factory = module.FreshnessSemanticVerifier

    class ModelSession:
        def __init__(self, fails):
            self.fails = fails

        def execute(self, _query):
            if self.fails and phase == "execute":
                raise module.psycopg.OperationalError("Model błędu kontrolnego SELECT")
            return self

        def fetchone(self):
            if self.fails and phase == "fetchone":
                raise module.psycopg.OperationalError("Model błędu odczytu SELECT")
            return ("freshness_crypto_verifier", "freshness_crypto_verifier", None, None, False)

        def close(self):
            with closes.get_lock():
                closes.value += 1

    def connect(**_coordinates):
        with connects.get_lock():
            connects.value += 1
            return ModelSession(fails=connects.value == 1)

    def factory(credentials, writer, identity):
        class SessionCheckedWriter:
            def prepare(self, binding, document, receipt):
                session = module.PostgreSQLVerifierConnection("/test/pg", 5432).connect()
                try:
                    return writer.prepare(binding, document, receipt)
                finally:
                    session.close()

        return original_factory(credentials, SessionCheckedWriter(), identity)

    monkeypatch.setattr(module.psycopg, "connect", connect)
    monkeypatch.setattr(module, "FreshnessSemanticVerifier", factory)
    with _server(tmp_path, module, requests=3) as server:
        runtime_tests._model_installed_coordinates(monkeypatch, server)
        authority, cas = _authority(candidate, server, monkeypatch)

        unavailable = authority.authenticate_and_advance(*server.inputs)

        assert unavailable.outcome is candidate.account.FreshnessAuthorityOutcome.UNAVAILABLE
        assert connects.value == closes.value == 1
        assert server.calls.value == 0
        cas.assert_not_called()
        accepted = authority.authenticate_and_advance(*server.inputs)
        assert accepted.outcome is candidate.account.FreshnessAuthorityOutcome.CAS_ACCEPTED
        assert connects.value == closes.value == 2
        assert server.calls.value == 1
        cas.assert_called_once()
        _finish_listener(server)


@pytest.mark.parametrize(
    "changes",
    [
        {"pid": 100000},
        {"uid": 100000},
        {"gid": 100000},
        {"start_time_ticks": 1},
        {"executable_sha256": "0" * 64},
    ],
)
def test_real_adapter_mismatched_expected_denies_before_payload(
    tmp_path, monkeypatch, candidate, changes
):
    """Odczyt tożsamości i socket są rzeczywiste; błędne expected dostarcza kontroler."""
    with runtime_tests._scripted(tmp_path) as server:
        runtime_tests._model_installed_coordinates(monkeypatch, server)
        approved = runtime_tests._record_expected(
            process_identity.capture_process_identity(server.process.pid)
        )
        authority, cas = _authority(
            candidate, server, monkeypatch, expected=replace(approved, **changes)
        )

        result = authority.authenticate_and_advance(b"{}", b"{}", b"{}")

        assert result.outcome is candidate.account.FreshnessAuthorityOutcome.UNAVAILABLE
        cas.assert_not_called()
        connections = 0 if {"uid", "gid"} & changes.keys() else 1
        assert runtime_tests._finish_count(server, connections=connections) == b""


@pytest.mark.parametrize("kind", ["cgroup", "socket", "exe-unreadable"])
def test_real_adapter_k4_missing_proof_denies_before_payload(
    tmp_path, monkeypatch, candidate, kind
):
    """Rzeczywisty transport; PermissionError odczytu exe to jawna iniekcja granicy."""
    with runtime_tests._scripted(tmp_path) as server:
        runtime_tests._model_installed_coordinates(monkeypatch, server)
        authority, cas = _authority(candidate, server, monkeypatch)
        if kind == "cgroup":
            monkeypatch.setattr(ipc, "CGROUP", "/system.slice/niezatwierdzona.service")
        elif kind == "socket":
            alias = tmp_path / "alias.sock"
            alias.symlink_to(server.address)
            monkeypatch.setattr(ipc, "VERIFIER_SOCKET_PATH", str(alias))
        else:
            real_open = process_identity.os.open

            def denied_exe(path, flags, *args, **kwargs):
                if path == "exe":
                    raise PermissionError("Model niedostępnego /proc/PID/exe")
                return real_open(path, flags, *args, **kwargs)

            monkeypatch.setattr(process_identity.os, "open", denied_exe)
        result = authority.authenticate_and_advance(b"{}", b"{}", b"{}")

        assert result.outcome is candidate.account.FreshnessAuthorityOutcome.UNAVAILABLE
        cas.assert_not_called()
        assert runtime_tests._finish_count(server, connections=0 if kind == "socket" else 1) == b""


def _prepared_response(verifier, inputs, field):
    prepared = verifier.verify_exact_candidate(*inputs)
    binding = parse_canonical_json(prepared.binding)
    # Hash PREPARED jest samospójny, ale nie pasuje do wskazanego obiektu.
    binding[field] = "0" * 64
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
    "kind",
    [
        "proposer_authentication_digest",
        "proposed_document_digest",
        "receipt_canonical_digest",
        "length",
        "json",
        "trailing",
    ],
)
def test_real_adapter_bad_response_is_unknown_without_cas_or_retry(
    tmp_path, monkeypatch, candidate, kind
):
    """Rzeczywiste ramki od skryptowego peer; brak semantycznego przygotowania PG."""
    values = fixture_candidate()
    inputs = tuple(canonical_json_bytes(value) for value in values[2:5])
    if kind.endswith("_digest"):
        response = runtime_tests._frame(_prepared_response(values[0], inputs, kind))
    elif kind == "length":
        response = struct.pack("!I", 2**32 - 1)
    elif kind == "json":
        response = runtime_tests._frame(b"{")
    else:
        response = runtime_tests._frame(b'{"outcome":"INVALID"}') + b"x"
    with runtime_tests._scripted(tmp_path, response) as server:
        runtime_tests._model_installed_coordinates(monkeypatch, server)
        authority, cas = _authority(candidate, server, monkeypatch)

        result = authority.authenticate_and_advance(*inputs)

        assert (
            result.outcome
            is candidate.account.FreshnessAuthorityOutcome.PREPARATION_OUTCOME_UNKNOWN
        )
        cas.assert_not_called()
        assert runtime_tests._finish_count(server)


@pytest.mark.parametrize("failure", ["partial-send", "broken-connection"])
def test_real_partial_send_is_unknown_with_one_connection(
    tmp_path, monkeypatch, candidate, failure
):
    """Pierwsze bajty rzeczywiście wysyła jądro; wyjątek sendall wstrzykuje model."""
    real_socket = socket.socket

    class PartialSocket(real_socket):
        def sendall(self, data, flags=0):
            assert self.send(data[:7], flags) > 0
            if failure == "partial-send":
                self.shutdown(socket.SHUT_WR)
                raise TimeoutError("Model timeout po rzeczywistym częściowym send")
            self.close()
            raise BrokenPipeError("Model zerwania po rzeczywistym częściowym send")

    with runtime_tests._scripted(tmp_path) as server:
        runtime_tests._model_installed_coordinates(monkeypatch, server)
        monkeypatch.setattr(
            ipc,
            "socket",
            SimpleNamespace(
                AF_UNIX=socket.AF_UNIX, SOCK_STREAM=socket.SOCK_STREAM, socket=PartialSocket
            ),
        )
        authority, cas = _authority(candidate, server, monkeypatch)

        result = authority.authenticate_and_advance(b"{}", b"{}", b"{}")

        assert (
            result.outcome
            is candidate.account.FreshnessAuthorityOutcome.PREPARATION_OUTCOME_UNKNOWN
        )
        cas.assert_not_called()
        assert runtime_tests._finish_count(server) == b'{"a'
