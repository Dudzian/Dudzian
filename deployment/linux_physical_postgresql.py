"""Odczytowa diagnostyka peer PostgreSQL bez prawa do kwalifikacji produktu.

Oddzielny proces sondy, nawet uruchomiony z UID usługi, nie jest tą usługą.
Adapter wewnątrz verifiera udostępnia częściowe, surowe obserwacje jego własnej
sesji. Pełne powiązanie obu usług runtime/verifier i semantyczna macierz
uprawnień pozostają BLOCKED niezależnie od wyniku osobnej sondy.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import PurePosixPath
from typing import Any, cast

ADAPTER_ID = "cryptohunter.linux.postgresql.service-connection.v1"
BLOCKERS = (
    "POSTGRESQL_RUNTIME_SERVICE_CONNECTION_ADAPTER_UNAVAILABLE",
    "POSTGRESQL_RUNTIME_PROCESS_BINDING_MISSING",
    "POSTGRESQL_RUNTIME_VERIFIER_PEER_PROOF_MISSING",
    "POSTGRESQL_PRODUCT_SEMANTIC_PERMISSION_MATRIX_UNPROVEN",
)
IDENTITY_SQL = (
    "SELECT system_user::text,session_user::text,current_user::text,pg_backend_pid(),"
    "inet_client_addr(),inet_server_addr(),current_setting('transaction_read_only'),"
    "coalesce((SELECT ssl FROM pg_stat_ssl WHERE pid=pg_backend_pid()),false)"
)
# LIMIT 0 sprawdza rzeczywistą odmowę SELECT bez odczytu haseł lub innych wierszy.
DENIAL_SQL = "SELECT 1 FROM pg_catalog.pg_authid LIMIT 0"


class PostgreSQLProbeError(ValueError):
    """Diagnostyka nie potwierdziła wymaganych, odczytowych faktów."""


class PostgreSQLQualificationBlocked(ValueError):
    """Brak niezależnego dowodu połączeń i uprawnień właściwych usług."""


def blocked_service_qualification() -> dict[str, Any]:
    """Zamknięty status implementacji; raport nie może zadeklarować adaptera."""
    return {
        "schema_version": 1,
        "adapter_id": ADAPTER_ID,
        "status": "BLOCKED",
        "blockers": list(BLOCKERS),
    }


def validate_blocked_service_qualification(value: object) -> None:
    """Sprawdź wyłącznie format jawnej odmowy; poprawny format nie daje PASS."""
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value != blocked_service_qualification()
    ):
        raise PostgreSQLProbeError("Nieprawidłowy status blokady kwalifikacji PostgreSQL usług")


def require_service_qualification(value: object) -> None:
    """Żaden dokument wejściowy nie zastępuje niezaimplementowanego adaptera.

    W szczególności administracyjne pg_settings/HBA/ident i osobny proces
    testowy z UID usługi nie dowodzą połączenia tej konkretnej usługi.
    """
    raise PostgreSQLQualificationBlocked(
        "BLOCKED: brak zatwierdzonego, związanego z procesem adaptera PostgreSQL właściwych usług"
    )


def validate_service_connection_observation(
    value: object,
    *,
    challenge_id: str,
    installation_id: str,
    role: str,
    expected_process: dict[str, Any],
    expected_os_principal: str,
    expected_database_role: str,
) -> None:
    """Odtwórz decyzję częściowego adaptera z surowych wierszy i faktów procesu.

    Wywołujący przypina expected_process z niezależnej obserwacji procesu usługi,
    nie z dokumentu odpowiedzi. Udany replay nie kwalifikuje brakującego runtime,
    nie ustanawia zaufania do nadawcy JSON ani nie zastępuje kontroli HBA/ident.
    """
    from bot_core.linux_service_postgresql import (
        ADAPTER_ID as SERVICE_ADAPTER_ID,
        ALLOWED_SQL,
        BACKEND_SQL,
        DENIAL_SQL as SERVICE_DENIAL_SQL,
        IDENTITY_SQL as SERVICE_IDENTITY_SQL,
        OPPOSITE_ROLE_SQL,
    )

    _require(
        type(value) is dict
        and set(value)
        == {
            "schema_version",
            "adapter_id",
            "scope",
            "challenge_id",
            "installation_id",
            "role",
            "process",
            "identity_sql",
            "identity_row",
            "backend_sql",
            "backend_row",
            "allowed",
            "denials",
            "transaction",
        }
        and type(value["schema_version"]) is int
        and value["schema_version"] == 1
        and value["adapter_id"] == SERVICE_ADAPTER_ID
        and value["scope"] == "SERVICE_OWNED_CONNECTION_OBSERVATION"
        and value["challenge_id"] == challenge_id
        and value["installation_id"] == installation_id
        and value["role"] == role
        and role in OPPOSITE_ROLE_SQL
        and all(
            type(v) is str and re.fullmatch(r"[0-9a-f]{64}", v)
            for v in (challenge_id, installation_id)
        )
        and _identifier(expected_os_principal)
        and _identifier(expected_database_role),
        "Nieprawidłowy kontekst obserwacji połączenia usługi",
    )
    report = cast(dict[str, Any], value)
    process = report["process"]
    _require(
        type(process) is dict
        and set(process)
        == {"pid", "uid", "gid", "groups", "start_time_ticks", "exe_sha256", "cgroup"}
        and process == expected_process
        and all(
            type(process[key]) is int and process[key] > 0
            for key in ("pid", "uid", "gid", "start_time_ticks")
        )
        and type(process["groups"]) is list
        and all(type(group) is int and group > 0 for group in process["groups"])
        and process["groups"] == sorted(set(process["groups"]))
        and type(process["exe_sha256"]) is str
        and re.fullmatch(r"[0-9a-f]{64}", process["exe_sha256"]) is not None
        and type(process["cgroup"]) is str
        and process["cgroup"].startswith("/")
        and "\n" not in process["cgroup"],
        "Proces odpowiedzi PostgreSQL różni się od niezależnie przypiętej usługi",
    )
    identity, backend = report["identity_row"], report["backend_row"]
    _require(
        type(identity) is list
        and len(identity) == 9
        and identity[:3]
        == ["peer:" + expected_os_principal, expected_database_role, expected_database_role]
        and type(identity[3]) is int
        and identity[3] > 0
        and identity[4:] == ["freshness_gate", None, None, "on", False]
        and type(identity[8]) is bool
        and type(backend) is list
        and len(backend) == 6
        and type(backend[0]) is int
        and backend
        == [identity[3], expected_database_role, "freshness_gate", None, -1, "client backend"]
        and type(backend[4]) is int
        and report["identity_sql"] == SERVICE_IDENTITY_SQL
        and report["backend_sql"] == BACKEND_SQL
        and report["allowed"] == {"sql": ALLOWED_SQL, "row": [1]}
        and type(report["allowed"]) is dict
        and type(report["allowed"]["row"]) is list
        and type(report["allowed"]["row"][0]) is int
        and report["denials"]
        == [
            {"sql": sql, "sqlstate": "42501"}
            for sql in (*SERVICE_DENIAL_SQL, OPPOSITE_ROLE_SQL[role])
        ]
        and report["transaction"] == "READ_ONLY_ROLLED_BACK",
        "Surowe obserwacje nie potwierdzają peer, mapowania, roli lub rzeczywistych odmów",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PostgreSQLProbeError(message)


def _identifier(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", value) is not None


def _process() -> dict[str, Any]:
    # Import lokalny pozwala NativeDriver dołączyć tę diagnostykę bez cyklu
    # importów. Funkcja czyta /proc; nie przyjmuje PID zadeklarowanego w JSON.
    from deployment.linux_physical_native import process_observation

    return cast(dict[str, Any], process_observation(os.getpid()))


def collect_role_diagnostic(
    config: dict[str, Any],
    *,
    role: str,
    expected_uid: int,
    expected_gid: int,
    expected_groups: list[int],
    expected_os_principal: str,
    expected_database_role: str,
    expected_executable_sha256: str,
    challenge_id: str,
) -> dict[str, Any]:
    """Zbierz rzeczywiste fakty własnego procesu, peer i odczytowej odmowy.

    Wywołujący uruchamia sondę osobno i jawnie z właściwą tożsamością. Funkcja
    nie zmienia UID/GID, nie używa SET ROLE, nie uruchamia usług i nie wykonuje
    SQL podanego przez operatora. Wynik zawsze pozostaje diagnostyczny BLOCKED.
    """
    if sys.platform != "linux":
        raise PostgreSQLProbeError("Sonda wymaga natywnego Linux")
    _require(
        isinstance(config, dict)
        and set(config) == {"socket", "port", "database"}
        and isinstance(config["socket"], str)
        and re.fullmatch(r"/[A-Za-z0-9_./-]{1,199}", config["socket"]) is not None
        and str(PurePosixPath(config["socket"])) == config["socket"]
        and all(part not in {".", ".."} for part in config["socket"].split("/")[1:])
        and type(config["port"]) is int
        and 1 <= config["port"] <= 65535
        and _identifier(config["database"]),
        "Nieprawidłowy lokalny endpoint PostgreSQL",
    )
    _require(
        isinstance(role, str)
        and role in {"runtime", "verifier"}
        and type(expected_uid) is int
        and expected_uid > 0
        and type(expected_gid) is int
        and expected_gid > 0
        and isinstance(expected_groups, list)
        and all(type(group) is int and group > 0 for group in expected_groups)
        and len(set(expected_groups)) == len(expected_groups)
        and _identifier(expected_os_principal)
        and _identifier(expected_database_role)
        and isinstance(expected_executable_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", expected_executable_sha256) is not None
        and isinstance(challenge_id, str)
        and re.fullmatch(r"[0-9a-f]{64}", challenge_id) is not None,
        "Nieprawidłowa przypięta tożsamość sondy",
    )
    # libpq pobiera brakujące ustawienia z otoczenia. W profilu peer żadna
    # odziedziczona zmienna PG* nie może wybrać hasła, usługi ani opcji sesji.
    # Sprawdzamy nazwy, bez odczytywania lub raportowania wartości sekretów.
    _require(
        not any(name.startswith("PG") for name in os.environ),
        "Niejawna konfiguracja libpq jest niedozwolona w sondzie peer",
    )
    _require(
        set(os.getresuid()) == {expected_uid}
        and set(os.getresgid()) == {expected_gid}
        and sorted(os.getgroups()) == sorted(expected_groups),
        "Sonda nie działa z dokładną nieadministracyjną tożsamością usługi",
    )
    import pwd

    try:
        principal = pwd.getpwuid(expected_uid)
    except KeyError:
        raise PostgreSQLProbeError("Nie można potwierdzić tożsamości konta systemowego") from None
    _require(
        principal.pw_name == expected_os_principal and principal.pw_uid == expected_uid,
        "Nazwa konta systemowego nie odpowiada przypiętemu UID",
    )
    before = _process()
    _require(
        before["pid"] == os.getpid()
        and before["uid"] == expected_uid
        and before["gid"] == expected_gid
        and sorted(before["groups"]) == sorted(expected_groups),
        "Tożsamość jądrowa procesu sondy jest niespójna",
    )
    _require(
        before["exe_sha256"] == expected_executable_sha256,
        "Program procesu sondy różni się od przypiętego programu",
    )

    import psycopg

    connection = None
    try:
        connection = psycopg.connect(
            host=config["socket"],
            port=config["port"],
            dbname=config["database"],
            user=expected_database_role,
            password=None,
            passfile="/dev/null",
            sslmode="disable",
            connect_timeout=5,
            application_name="cryptohunter-stage9-readonly-role-probe",
            options=(
                "-c default_transaction_read_only=on -c statement_timeout=5000 "
                "-c lock_timeout=1000 -c idle_in_transaction_session_timeout=5000"
            ),
        )
        identity = connection.execute(IDENTITY_SQL).fetchone()
        _require(
            isinstance(identity, tuple)
            and len(identity) == 8
            and identity[0] == "peer:" + expected_os_principal
            and identity[1] == identity[2] == expected_database_role
            and type(identity[3]) is int
            and identity[3] > 0
            and identity[4:6] == (None, None)
            and identity[6] == "on"
            and identity[7] is False,
            "Serwer nie potwierdził lokalnej, uwierzytelnionej tożsamości peer",
        )
        denied = False
        try:
            # Zagnieżdżona transakcja psycopg tworzy SAVEPOINT, dzięki czemu
            # spodziewana odmowa nie ukrywa błędów w pozostałych odczytach.
            with connection.transaction():
                connection.execute(DENIAL_SQL)
        except psycopg.Error as exc:
            _require(exc.sqlstate == "42501", "Odmowa nie wynika z uprawnień PostgreSQL")
            denied = True
        _require(denied, "Nieautoryzowany odczyt nie został odrzucony")
        after = _process()
        _require(after == before, "Proces sondy zmienił się podczas połączenia")
        return {
            "schema_version": 1,
            "challenge_id": challenge_id,
            "role": role,
            "scope": "ROLE_PROBE_DIAGNOSTIC_ONLY",
            "status": "BLOCKED",
            "process": before,
            "connection": {
                "system_user": identity[0],
                "session_user": identity[1],
                "current_user": identity[2],
                "backend_pid": identity[3],
                "client_addr": None,
                "server_addr": None,
                "transaction_read_only": "on",
                "ssl": False,
            },
            "denial": {"operation": "READ_AUTHID_CATALOG_WITHOUT_ROWS", "sqlstate": "42501"},
            "service_qualification": blocked_service_qualification(),
        }
    except PostgreSQLProbeError:
        raise
    except (psycopg.Error, OSError, ValueError, KeyError, TypeError):
        raise PostgreSQLProbeError(
            "Nie można bezpiecznie zebrać diagnostyki peer PostgreSQL"
        ) from None
    finally:
        if connection is not None:
            try:
                connection.rollback()
            finally:
                connection.close()
