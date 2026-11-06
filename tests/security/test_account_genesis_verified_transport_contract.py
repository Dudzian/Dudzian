"""TEST_ONLY: mapowanie klienta i niezmienione CAS z jawnym modelem PostgreSQL.

Nie ma tu dowodu AF_UNIX, instalacji, niezależnego zatwierdzenia expected ani
natywnego rozdzielenia dwóch UID. Test transportu bada te granice oddzielnie.
Model zwracanego PREPARED powstaje z prawdziwej weryfikacji podpisów fixture,
ale przygotowanie i wszystkie połączenia SQL są modelami bez trwałości.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import psycopg
import pytest

from bot_core import account_genesis_freshness_authority as original, freshness_runtime_ipc as ipc
from bot_core.freshness_verifier_service import VERIFIER_SOCKET_PATH
from bot_core.postgresql_freshness_authority import canonical_json_bytes, parse_canonical_json
from deployment.linux_physical_ipc import ExpectedPeerProcess
from tests.security.account_genesis_transport_review_support import load_account_genesis
from tests.security.test_freshness_semantic_verifier import fixture_candidate


@pytest.fixture
def module(tmp_path):
    return load_account_genesis(tmp_path)


@pytest.fixture
def authority(module, tmp_path):
    # Jawny model zatwierdzonych oczekiwań; żaden serwer nie dostarcza tych danych.
    expected = ExpectedPeerProcess(101, 10001, 10002, 1234, "a" * 64)
    config = module.ProductionLocalFreshnessAuthorityConfig(
        VERIFIER_SOCKET_PATH, str(tmp_path), 5432
    )
    return module.ProductionLocalFreshnessAuthority(config, expected_verifier=expected)


@pytest.fixture
def prepared():
    values = fixture_candidate()
    inputs = tuple(canonical_json_bytes(value) for value in values[2:5])
    candidate = values[0].verify_exact_candidate(*inputs)
    binding = candidate.binding
    preparation_id = parse_canonical_json(binding)["preparation_id"]
    return inputs, ipc.FreshnessIPCResult(ipc.FreshnessIPCOutcome.PREPARED, preparation_id, binding)


def _exchange(monkeypatch, authority, result):
    operation = Mock(return_value=result)
    monkeypatch.setattr(authority._verifier_client, "verify_exact_candidate", operation)
    return operation


def _forbid_sql(monkeypatch):
    connection = Mock(side_effect=AssertionError("Próba SQL bez poprawnego PREPARED"))
    monkeypatch.setattr(psycopg, "connect", connection)
    return connection


@pytest.fixture
def linux_client_before_transport(monkeypatch):
    """Model platformy dla lokalnych limitów; żadnego socketu ani dowodu IPC."""
    constructor = Mock(side_effect=AssertionError("Kontrola lokalna nie może tworzyć socketu"))
    monkeypatch.setattr(ipc, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(ipc, "socket", SimpleNamespace(AF_UNIX=object(), socket=constructor))
    monkeypatch.setattr(ipc, "_require_runtime_identity", lambda _expected: None)
    yield
    constructor.assert_not_called()


@pytest.mark.parametrize(
    "outcome,reason,expected",
    [
        (ipc.FreshnessIPCOutcome.INVALID, "VERIFIER_RESPONSE", "INVALID_AUTHENTICATION"),
        (ipc.FreshnessIPCOutcome.INVALID, "INVALID_LOCAL_INPUT", "INVALID_DOCUMENT"),
        (ipc.FreshnessIPCOutcome.INVALID, "REQUEST_TOO_LARGE", "INVALID_DOCUMENT"),
        (ipc.FreshnessIPCOutcome.UNAVAILABLE, "TRANSPORT_OR_IDENTITY_UNAVAILABLE", "UNAVAILABLE"),
        (ipc.FreshnessIPCOutcome.UNAVAILABLE, "VERIFIER_RESPONSE", "UNAVAILABLE"),
        (
            ipc.FreshnessIPCOutcome.OUTCOME_UNKNOWN,
            "VERIFIER_RESPONSE",
            "PREPARATION_OUTCOME_UNKNOWN",
        ),
        (
            ipc.FreshnessIPCOutcome.OUTCOME_UNKNOWN,
            "AMBIGUOUS_EXCHANGE",
            "PREPARATION_OUTCOME_UNKNOWN",
        ),
    ],
)
def test_client_results_map_once_without_sql_or_retry(
    authority, monkeypatch, outcome, reason, expected
):
    exchange = _exchange(monkeypatch, authority, ipc.FreshnessIPCResult(outcome, reason=reason))
    sql = _forbid_sql(monkeypatch)
    result = authority.authenticate_and_advance(b"proposer", b"document", b"receipt")
    assert result.outcome.value == expected
    exchange.assert_called_once_with(b"proposer", b"document", b"receipt")
    sql.assert_not_called()


@pytest.mark.parametrize("position", range(3))
@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), "{}", bytearray(b"{}")])
def test_non_bytes_input_never_reaches_client_or_sql(authority, monkeypatch, position, invalid):
    exchange = _exchange(monkeypatch, authority, None)
    sql = _forbid_sql(monkeypatch)
    inputs = [b"p", b"d", b"r"]
    inputs[position] = invalid
    result = authority.authenticate_and_advance(*inputs)
    assert result.outcome.value == "INVALID_DOCUMENT"
    exchange.assert_not_called()
    sql.assert_not_called()


@pytest.mark.parametrize("position", range(3))
@pytest.mark.parametrize(
    "invalid", [b"", b"x" * (ipc.MAX_FRAME_BYTES + 1)], ids=["empty", "oversized"]
)
def test_local_length_refusal_uses_actual_client_without_transport_or_sql(
    authority, monkeypatch, linux_client_before_transport, position, invalid
):
    # Platforma/NSS/UID są modelem; kontrola rozmiaru i mapowanie to prawdziwy kod.
    transport = Mock(side_effect=AssertionError("Nie wolno łączyć przy złym rozmiarze"))
    monkeypatch.setattr(ipc, "_pinned_verifier_socket", transport)
    sql = _forbid_sql(monkeypatch)
    inputs = [b"p", b"d", b"r"]
    inputs[position] = invalid
    assert authority.authenticate_and_advance(*inputs).outcome.value == "INVALID_DOCUMENT"
    transport.assert_not_called()
    sql.assert_not_called()


def test_encoded_request_limit_refuses_before_transport_or_sql(
    authority, monkeypatch, linux_client_before_transport
):
    transport = Mock(side_effect=AssertionError("Nie wolno wysyłać za dużej ramki"))
    monkeypatch.setattr(ipc, "_pinned_verifier_socket", transport)
    sql = _forbid_sql(monkeypatch)
    assert authority.authenticate_and_advance(*(b"x" * 1100000,) * 3).outcome.value == (
        "INVALID_DOCUMENT"
    )
    transport.assert_not_called()
    sql.assert_not_called()


def test_missing_af_unix_refuses_before_identity_transport_or_sql(authority, monkeypatch):
    """Brak transportu nadal odmawia, także w modelu platformy Linux."""
    monkeypatch.setattr(ipc, "sys", SimpleNamespace(platform="linux"))
    constructor = Mock(side_effect=AssertionError("Brak AF_UNIX musi poprzedzać socket"))
    monkeypatch.setattr(ipc, "socket", SimpleNamespace(socket=constructor))
    identity = Mock(side_effect=AssertionError("Brak AF_UNIX musi poprzedzać tożsamość"))
    transport = Mock(side_effect=AssertionError("Brak AF_UNIX musi poprzedzać transport"))
    monkeypatch.setattr(ipc, "_require_runtime_identity", identity)
    monkeypatch.setattr(ipc, "_pinned_verifier_socket", transport)
    sql = _forbid_sql(monkeypatch)
    assert authority.authenticate_and_advance(b"p", b"d", b"r").outcome.value == "UNAVAILABLE"
    identity.assert_not_called()
    transport.assert_not_called()
    constructor.assert_not_called()
    sql.assert_not_called()


@pytest.mark.parametrize(
    "preparation_id,binding",
    [
        (None, None),
        ("a" * 64, None),
        ("wrong", b"{}"),
        ("a" * 64, b"{}"),
        ("a" * 64, b"not json"),
        ("a" * 64, b"[]"),
        ("a" * 64, b'{"preparation_id":"other"}'),
    ],
)
def test_incomplete_prepared_cannot_start_sql(authority, monkeypatch, preparation_id, binding):
    exchange = _exchange(
        monkeypatch,
        authority,
        ipc.FreshnessIPCResult(ipc.FreshnessIPCOutcome.PREPARED, preparation_id, binding),
    )
    sql = _forbid_sql(monkeypatch)
    assert authority.authenticate_and_advance(b"p", b"d", b"r").outcome.value == (
        "PREPARATION_OUTCOME_UNKNOWN"
    )
    exchange.assert_called_once()
    sql.assert_not_called()


def test_independently_supplied_expected_is_mandatory_and_exact(module, tmp_path):
    config = module.ProductionLocalFreshnessAuthorityConfig(
        VERIFIER_SOCKET_PATH, str(tmp_path), 5432
    )
    with pytest.raises(TypeError):
        module.ProductionLocalFreshnessAuthority(config)
    for expected in (None, {}, object()):
        with pytest.raises(TypeError, match="EXACT_APPROVED_VERIFIER_PROCESS_REQUIRED"):
            module.ProductionLocalFreshnessAuthority(config, expected_verifier=expected)


def test_alternate_socket_is_refused_without_fallback(module, tmp_path):
    config = module.ProductionLocalFreshnessAuthorityConfig(
        str(tmp_path / "unverified.sock"), str(tmp_path), 5432
    )
    with pytest.raises(ValueError, match="fixed protected verifier socket required"):
        module.ProductionLocalFreshnessAuthority(
            config, expected_verifier=ExpectedPeerProcess(101, 10001, 10002, 1234, "a" * 64)
        )


def test_migration_preserves_public_operation_and_exact_cas_methods(module):
    assert {
        name for name in dir(module.ProductionLocalFreshnessAuthority) if not name.startswith("_")
    } == {"authenticate_and_advance"}
    for method in (
        "authenticate_and_advance",
        "_connect_runtime",
        "_compare_and_advance",
        "_compare_and_advance_once",
    ):
        assert inspect.getsource(getattr(module.ProductionLocalFreshnessAuthority, method)) == (
            inspect.getsource(getattr(original.ProductionLocalFreshnessAuthority, method))
        )
    source = Path(module.__file__).read_text()
    assert "socket.socket" not in source
    assert "_request_preparation_once" not in source
    assert "_PreparationTransportState" not in source


@pytest.mark.parametrize("accepted", ["CAS_ACCEPTED", "ALREADY_ACCEPTED_EXACT"])
def test_prepared_starts_separate_existing_serializable_cas(
    authority, prepared, monkeypatch, accepted
):
    inputs, response = prepared
    exchange = _exchange(monkeypatch, authority, response)
    connection = Mock()
    connection.execute.return_value.fetchone.return_value = (
        accepted,
        7,
        parse_canonical_json(inputs[2]),
    )
    connect = Mock(return_value=connection)
    monkeypatch.setattr(authority, "_connect_runtime", connect)
    result = authority.authenticate_and_advance(*inputs)
    assert result.outcome.value == accepted
    assert result.decision_sequence == 7
    assert result.retained_receipt == inputs[2]
    assert result.preparation_id == response.preparation_id
    exchange.assert_called_once_with(*inputs)
    connect.assert_called_once_with()
    calls = connection.execute.call_args_list
    assert calls[0].args == ("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE",)
    assert calls[1].args[0] == (
        "SELECT * FROM freshness_authority.compare_and_advance(%s,%s,%s,%s)"
    )
    identifier, binding, document, receipt = calls[1].args[1]
    assert identifier == response.preparation_id
    assert binding.obj == parse_canonical_json(response.binding)
    assert (document, receipt) == inputs[1:]
    connection.commit.assert_called_once_with()
    connection.rollback.assert_not_called()
    connection.close.assert_called_once_with()


def test_runtime_connection_keeps_exact_peer_principal(authority, monkeypatch):
    connection = Mock()
    connection.execute.return_value.fetchone.return_value = (
        "freshness_runtime",
        "freshness_runtime",
        None,
        None,
        False,
    )
    connect = Mock(return_value=connection)
    monkeypatch.setattr(psycopg, "connect", connect)
    assert authority._connect_runtime() is connection
    connect.assert_called_once_with(
        host=authority._config.postgres_socket_directory,
        port=5432,
        dbname="freshness_gate",
        user="freshness_runtime",
        sslmode="disable",
    )
    connection.rollback.assert_called_once_with()
    connection.commit.assert_not_called()
    connection.close.assert_not_called()


@pytest.mark.parametrize("error", [psycopg.OperationalError, psycopg.InterfaceError])
def test_ambiguous_cas_commit_is_not_a_preparation_retry(authority, prepared, monkeypatch, error):
    inputs, response = prepared
    exchange = _exchange(monkeypatch, authority, response)
    connection = Mock()
    connection.commit.side_effect = error("Model utraty odpowiedzi COMMIT")
    connect = Mock(return_value=connection)
    monkeypatch.setattr(authority, "_connect_runtime", connect)
    result = authority.authenticate_and_advance(*inputs)
    assert result.outcome.value == "FRESHNESS_OUTCOME_UNKNOWN"
    exchange.assert_called_once()
    connect.assert_called_once()
    connection.commit.assert_called_once()
    connection.rollback.assert_called_once()
    connection.close.assert_called_once()


@pytest.mark.parametrize("failures", [1, 3])
def test_native_cas_serialization_retry_never_repeats_preparation(
    authority, prepared, monkeypatch, failures
):
    inputs, response = prepared
    exchange = _exchange(monkeypatch, authority, response)
    connections = [Mock() for _ in range(min(failures + 1, 3))]
    for index, connection in enumerate(connections):
        connection.execute.return_value.fetchone.return_value = (
            "CAS_ACCEPTED",
            8,
            parse_canonical_json(inputs[2]),
        )
        if index < failures:
            connection.commit.side_effect = psycopg.errors.SerializationFailure("Model 40001")
    connect = Mock(side_effect=connections)
    monkeypatch.setattr(authority, "_connect_runtime", connect)
    monkeypatch.setattr(original.time, "sleep", lambda _seconds: None)
    result = authority.authenticate_and_advance(*inputs)
    expected = "CAS_ACCEPTED" if failures == 1 else "FRESHNESS_OUTCOME_UNKNOWN"
    assert result.outcome.value == expected
    assert connect.call_count == len(connections)
    exchange.assert_called_once_with(*inputs)
    expected_arguments = connections[0].execute.call_args_list[1].args[1]
    for connection in connections:
        arguments = connection.execute.call_args_list[1].args[1]
        assert arguments[0] == expected_arguments[0]
        assert arguments[1].obj == expected_arguments[1].obj
        assert arguments[2:] == expected_arguments[2:]
        connection.close.assert_called_once()
