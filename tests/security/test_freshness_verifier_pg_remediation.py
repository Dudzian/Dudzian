"""Model granicy psycopg: własność sesji podczas kontroli lokalnego połączenia."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.security.freshness_verifier_remediation_support import load_verifier

_VALID_SESSION_FACTS = (
    "freshness_crypto_verifier",
    "freshness_crypto_verifier",
    None,
    None,
    False,
)


@pytest.fixture
def verifier(tmp_path):
    """Załaduj wybrany wariant bez zmiany podpisanego pliku produkcyjnego."""
    return load_verifier(tmp_path)


@pytest.fixture
def pg_boundary(verifier, monkeypatch):
    """Zastąp wyłącznie granicę psycopg; nie otwieraj prawdziwej sesji PG."""
    connection = Mock(spec=["execute", "close"])
    cursor = Mock(spec=["fetchone"])
    connection.execute.return_value = cursor
    cursor.fetchone.return_value = _VALID_SESSION_FACTS
    connect = Mock(return_value=connection)
    monkeypatch.setattr(verifier.psycopg, "connect", connect)
    return SimpleNamespace(
        module=verifier,
        connection=connection,
        cursor=cursor,
        connect=connect,
        configuration=verifier.PostgreSQLVerifierConnection("/test/pg", 5432),
    )


@pytest.mark.parametrize("phase", ["execute", "fetchone"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_session_check_error_closes_once_and_preserves_primary(pg_boundary, phase, cleanup_fails):
    """Awaria SELECT lub odczytu wyniku zamyka sesję i zachowuje przyczynę odmowy."""
    boundary = pg_boundary
    primary = boundary.module.psycopg.OperationalError("pierwotna awaria kontroli sesji")
    operation = boundary.connection.execute if phase == "execute" else boundary.cursor.fetchone
    operation.side_effect = primary
    if cleanup_fails:
        boundary.connection.close.side_effect = boundary.module.psycopg.Error(
            "wtórna awaria zamykania"
        )

    with pytest.raises(
        boundary.module.FreshnessVerifierUnavailable,
        match="verifier database session unavailable",
    ) as caught:
        boundary.configuration.connect()

    assert caught.value.__cause__ is primary, "Cleanup nie może nadpisać przyczyny awarii SELECT"
    boundary.connection.close.assert_called_once_with()
    boundary.connection.execute.assert_called_once()
    if phase == "execute":
        boundary.cursor.fetchone.assert_not_called()
    else:
        boundary.cursor.fetchone.assert_called_once_with()


@pytest.mark.parametrize(
    "facts",
    [
        ("obca_rola", "freshness_crypto_verifier", None, None, False),
        ("freshness_crypto_verifier", "obca_rola", None, None, False),
        ("freshness_crypto_verifier", "freshness_crypto_verifier", "127.0.0.1", None, False),
        ("freshness_crypto_verifier", "freshness_crypto_verifier", None, "127.0.0.1", False),
        ("freshness_crypto_verifier", "freshness_crypto_verifier", None, None, True),
        None,
    ],
    ids=["session-user", "current-user", "client-address", "server-address", "ssl", "no-row"],
)
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_wrong_session_facts_close_once_even_if_cleanup_fails(pg_boundary, facts, cleanup_fails):
    """Każda niezgodna własność sesji powoduje odmowę i dokładnie jedną próbę close."""
    boundary = pg_boundary
    boundary.cursor.fetchone.return_value = facts
    if cleanup_fails:
        boundary.connection.close.side_effect = boundary.module.psycopg.Error(
            "wtórna awaria zamykania odrzuconej sesji"
        )

    with pytest.raises(
        boundary.module.FreshnessVerifierUnavailable,
        match="non-peer or wrong database session",
    ):
        boundary.configuration.connect()

    boundary.connection.close.assert_called_once_with()
    boundary.connection.execute.assert_called_once()
    boundary.cursor.fetchone.assert_called_once_with()


def test_valid_session_stays_open_until_caller_closes(pg_boundary):
    """Poprawne połączenie przekazuje własność wywołującemu bez przedwczesnego close."""
    boundary = pg_boundary

    returned = boundary.configuration.connect()

    assert returned is boundary.connection, "Wywołujący musi otrzymać sprawdzoną sesję"
    boundary.connection.close.assert_not_called()
    boundary.connect.assert_called_once_with(
        host="/test/pg",
        port=5432,
        dbname="freshness_gate",
        user="freshness_crypto_verifier",
        sslmode="disable",
    )
    boundary.connection.execute.assert_called_once()
    boundary.cursor.fetchone.assert_called_once_with()
    returned.close()
    boundary.connection.close.assert_called_once_with()


@pytest.mark.parametrize("phase", ["execute", "fetchone"])
def test_unexpected_select_error_still_closes_once(pg_boundary, phase):
    boundary = pg_boundary
    primary = RuntimeError("nieoczekiwany błąd granicy SELECT")
    operation = boundary.connection.execute if phase == "execute" else boundary.cursor.fetchone
    operation.side_effect = primary
    with pytest.raises(RuntimeError) as caught:
        boundary.configuration.connect()
    assert caught.value is primary
    boundary.connection.close.assert_called_once_with()


@pytest.mark.parametrize("error_name", ["Error", "OperationalError"])
def test_connect_failure_preserves_cause_without_closing_unacquired_session(
    pg_boundary, error_name
):
    """Awaria connect odmawia bez próby zamknięcia nieuzyskanej sesji."""
    boundary = pg_boundary
    primary = getattr(boundary.module.psycopg, error_name)("połączenie lokalne niedostępne")
    boundary.connect.side_effect = primary

    with pytest.raises(
        boundary.module.FreshnessVerifierUnavailable,
        match="verifier database session unavailable",
    ) as caught:
        boundary.configuration.connect()

    assert caught.value.__cause__ is primary, "Przyczyna awarii connect musi pozostać dostępna"
    boundary.connect.assert_called_once()
    boundary.connection.execute.assert_not_called()
    boundary.cursor.fetchone.assert_not_called()
    boundary.connection.close.assert_not_called()
