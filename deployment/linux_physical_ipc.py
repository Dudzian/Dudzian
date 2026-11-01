"""Odczytowy adapter rzeczywistego verifiera i jądrowe obserwacje IPC.

Stała diagnostyka AF_UNIX nie udostępnia podpisywania ani przygotowania
kandydata. Brak rzeczywistej usługi runtime i jej macierzy odmów nadal
blokuje pełny kontrakt produktu.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import os
from pathlib import Path
import re
import socket
import stat
import struct
import sys
import time
from typing import Any, cast

IPC_CONTRACT = "cryptohunter-linux-ipc-security-v1"
IPC_BLOCKERS = (
    "RUNTIME_IPC_APPLICATION_ADAPTER_UNAVAILABLE",
    "ACTUAL_RUNTIME_CLIENT_EXCHANGE_UNPROVEN",
    "RUNTIME_UNAUTHORIZED_IDENTITY_DENIAL_UNPROVEN",
    "DURABLE_CHALLENGE_REPLAY_PROTECTION_UNAVAILABLE",
)


class LinuxIPCObservationError(ValueError):
    """Nie można powiązać socketu z dokładnym procesem."""


class LinuxIPCQualificationBlocked(LinuxIPCObservationError):
    """Istniejąca architektura nie udostępnia kompletnego testu produktu."""


@dataclass(frozen=True)
class KernelPeerCredentials:
    """Wartości wyłącznie z SO_PEERCRED danego połączenia."""

    pid: int
    uid: int
    gid: int

    def __post_init__(self) -> None:
        if (
            type(self.pid) is not int
            or self.pid <= 0
            or any(type(value) is not int or value < 0 for value in (self.uid, self.gid))
        ):
            raise LinuxIPCObservationError("Nieprawidłowa jądrowa tożsamość partnera IPC")


@dataclass(frozen=True)
class ExpectedPeerProcess:
    """Proces przypięty niezależnie przez obserwację cgroup i zatwierdzone wydanie."""

    pid: int
    uid: int
    gid: int
    start_time_ticks: int
    executable_sha256: str

    def __post_init__(self) -> None:
        KernelPeerCredentials(self.pid, self.uid, self.gid)
        if (
            type(self.start_time_ticks) is not int
            or self.start_time_ticks <= 0
            or type(self.executable_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", self.executable_sha256) is None
        ):
            raise LinuxIPCObservationError("Brak dokładnej tożsamości programu partnera IPC")


def blocked_ipc_qualification() -> dict[str, Any]:
    """Jawna diagnoza brakującego adaptera; nigdy nie jest dowodem wykonania."""
    return {"contract": IPC_CONTRACT, "status": "BLOCKED", "blockers": list(IPC_BLOCKERS)}


def validate_blocked_ipc_qualification(value: object) -> None:
    """Walidacja formatu diagnozy nie przyznaje żadnej kwalifikacji."""
    if (
        type(value) is not dict
        or set(value) != {"contract", "status", "blockers"}
        or type(value["contract"]) is not str
        or type(value["status"]) is not str
        or type(value["blockers"]) is not list
        or value != blocked_ipc_qualification()
    ):
        raise LinuxIPCQualificationBlocked("Niezweryfikowana deklaracja kwalifikacji IPC")


def require_ipc_qualification(value: object) -> None:
    """Żaden raport nie zastąpi niezaimplementowanego adaptera produktu."""
    validate_blocked_ipc_qualification(value)
    raise LinuxIPCQualificationBlocked("Brak funkcjonalnego dowodu autoryzacji IPC produktu")


def observe_kernel_peer(peer: socket.socket) -> KernelPeerCredentials:
    """Odczytaj fakty połączenia; nie uznawaj ich za decyzję serwera aplikacji."""
    if sys.platform != "linux":
        raise LinuxIPCObservationError("SO_PEERCRED wymaga natywnego Linux")
    if (
        not isinstance(peer, socket.socket)
        or peer.family != socket.AF_UNIX
        or peer.type != socket.SOCK_STREAM
    ):
        raise LinuxIPCObservationError("Wymagane połączenie strumieniowe AF_UNIX")
    try:
        raw = peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        return KernelPeerCredentials(*struct.unpack("3i", raw))
    except (OSError, ValueError, struct.error) as exc:
        raise LinuxIPCObservationError("Nie można odczytać jądrowej tożsamości IPC") from exc


def observe_pinned_peer(peer: socket.socket, expected: ExpectedPeerProcess) -> dict[str, Any]:
    """Odczytowa obserwacja dokładnego procesu z uchwytem związanym z socketem.

    API wymaga już zestawionego połączenia i nie dodaje protokołu produktu.
    Tożsamość klienta znana serwerowi wymaga osobnej obserwacji po stronie
    rzeczywistej aplikacji; wynik tego API pozostaje niekwalifikujący.
    """
    credentials = observe_kernel_peer(peer)
    if credentials != KernelPeerCredentials(expected.pid, expected.uid, expected.gid):
        raise LinuxIPCObservationError("Partner IPC różni się od przypiętego procesu")
    if credentials.pid == os.getpid():
        raise LinuxIPCObservationError("Partner IPC musi być odrębnym procesem")
    # Używamy istniejącej kontroli SO_PEERPIDFD. pidfd_open po samym PID nie
    # dowodzi, że odziedziczony socket nadal należy do pierwotnego procesu.
    from bot_core.production_signer_ipc import ProductionSignerIPCError, _PinnedPeer
    from deployment.linux_physical_native import process_observation

    try:
        with _PinnedPeer(peer, expected.uid, expected.gid) as pinned:
            first = process_observation(expected.pid)
            if (
                first["pid"] != expected.pid
                or first["uid"] != expected.uid
                or first["gid"] != expected.gid
                or first["start_time_ticks"] != expected.start_time_ticks
                or first["exe_sha256"] != expected.executable_sha256
            ):
                raise LinuxIPCObservationError(
                    "Program lub okres życia partnera IPC jest niezgodny"
                )
            pinned.recheck()
            second = process_observation(expected.pid)
            if second != first:
                raise LinuxIPCObservationError("Partner IPC zmienił się podczas obserwacji")
            pinned.recheck()
            return {
                "contract": IPC_CONTRACT,
                "status": "UNQUALIFIED_TRANSPORT_OBSERVATION",
                "production_qualification": "BLOCKED",
                "peer_credentials": {
                    "pid": credentials.pid,
                    "uid": credentials.uid,
                    "gid": credentials.gid,
                },
                "peer_process": first,
                "socket_bound_pidfd": True,
            }
    except (ProductionSignerIPCError, OSError, ValueError, KeyError, IndexError) as exc:
        raise LinuxIPCObservationError("Nie można niezależnie przypiąć partnera IPC") from exc


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise LinuxIPCObservationError(message)


def _process_record(value: object, *, missing_executable: bool = False) -> dict[str, Any]:
    _check(
        type(value) is dict
        and set(value)
        == {"pid", "uid", "gid", "groups", "start_time_ticks", "exe_sha256", "cgroup"},
        "Niekompletna surowa tożsamość procesu IPC",
    )
    record = cast(dict[str, Any], value)
    _check(
        all(type(record[key]) is int and record[key] > 0 for key in ("pid", "start_time_ticks"))
        and all(type(record[key]) is int and record[key] >= 0 for key in ("uid", "gid"))
        and type(record["groups"]) is list
        and all(type(group) is int and group >= 0 for group in record["groups"])
        and record["groups"] == sorted(set(record["groups"]))
        and (
            (missing_executable and record["exe_sha256"] is None)
            or (
                type(record["exe_sha256"]) is str
                and re.fullmatch(r"[0-9a-f]{64}", record["exe_sha256"]) is not None
            )
        )
        and type(record["cgroup"]) is str
        and record["cgroup"].startswith("/")
        and not any(character in record["cgroup"] for character in "\x00\r\n"),
        "Nieprawidłowa surowa tożsamość procesu IPC",
    )
    return record


def _expected_process(process: dict[str, Any], expected: ExpectedPeerProcess, cgroup: str) -> None:
    _check(
        process["pid"] == expected.pid
        and process["uid"] == expected.uid
        and process["gid"] == expected.gid
        and process["start_time_ticks"] == expected.start_time_ticks
        and process["exe_sha256"] == expected.executable_sha256
        and process["cgroup"] == cgroup,
        "Odpowiedź nie należy do przypiętego procesu verifiera",
    )


def validate_verifier_observation(
    value: object,
    *,
    expected: ExpectedPeerProcess,
    expected_cgroup: str,
    challenge_id: str,
    installation_id: str,
) -> None:
    """Odtwórz częściowy exchange z surowych obu tożsamości i odpowiedzi.

    Zaufanie do podpisanego producenta i oczekiwanego procesu ustanawia
    wywołujący. Sam JSON nie jest dowodem AF_UNIX ani kwalifikacją produktu.
    """
    from bot_core.linux_service_ipc import PARTIAL_BLOCKERS, READONLY_OPERATION

    _check(
        all(
            type(item) is str and re.fullmatch(r"[0-9a-f]{64}", item) is not None
            for item in (challenge_id, installation_id)
        )
        and type(expected_cgroup) is str
        and expected_cgroup.startswith("/")
        and not any(character in expected_cgroup for character in "\x00\r\n"),
        "Brak przypiętego kontekstu częściowej obserwacji IPC",
    )
    _check(
        type(value) is dict
        and set(value)
        == {
            "contract",
            "status",
            "production_qualification",
            "request",
            "response",
            "client_process",
            "server_peer_credentials",
            "server_process",
            "socket_bound_pidfd",
        }
        and value["contract"] == IPC_CONTRACT
        and value["status"] == "PARTIAL_VERIFIER_OBSERVATION"
        and value["production_qualification"] == "BLOCKED"
        and value["socket_bound_pidfd"] is True,
        "Nieprawidłowy format częściowej obserwacji verifiera",
    )
    value = cast(dict[str, Any], value)
    request = value["request"]
    response = value["response"]
    _check(
        type(request) is dict
        and set(request) == {"version", "operation", "installation_id", "challenge_id", "nonce"}
        and type(request["version"]) is int
        and request["version"] == 1
        and request["operation"] == READONLY_OPERATION
        and request["installation_id"] == installation_id
        and request["challenge_id"] == challenge_id
        and type(request["nonce"]) is str
        and re.fullmatch(r"[0-9a-f]{64}", request["nonce"]) is not None,
        "Podmieniony challenge lub zakres żądania verifiera",
    )
    _check(
        type(response) is dict
        and set(response)
        == set(request)
        | {
            "status",
            "production_qualification",
            "server_process",
            "server_observed_client",
            "socket_bound_pidfd",
            "postgresql",
            "blockers",
        }
        and all(
            type(response[key]) is type(item) and response[key] == item
            for key, item in request.items()
        )
        and response["status"] == "PARTIAL_VERIFIER_OBSERVATION"
        and response["production_qualification"] == "BLOCKED"
        and response["socket_bound_pidfd"] is True,
        "Podmieniona odpowiedź aplikacji verifiera",
    )
    server = _process_record(value["server_process"])
    _expected_process(server, expected, expected_cgroup)
    _check(response["server_process"] == server, "Niespójny proces odpowiadającej aplikacji")
    server_credentials = {key: server[key] for key in ("pid", "uid", "gid")}
    _check(
        type(value["server_peer_credentials"]) is dict
        and value["server_peer_credentials"] == server_credentials
        and all(type(item) is int for item in value["server_peer_credentials"].values()),
        "SO_PEERCRED nie potwierdza procesu verifiera",
    )
    client = _process_record(value["client_process"])
    _check(client["uid"] == client["gid"] == 0, "Klient diagnostyczny nie jest root")
    observed = response["server_observed_client"]
    _check(
        type(observed) is dict and set(observed) == {"peer_credentials", "process"},
        "Brak obserwacji klienta po stronie aplikacji",
    )
    observed_process = _process_record(observed["process"], missing_executable=True)
    client_credentials = {key: client[key] for key in ("pid", "uid", "gid")}
    _check(
        type(observed["peer_credentials"]) is dict
        and observed["peer_credentials"] == client_credentials
        and all(type(item) is int for item in observed["peer_credentials"].values())
        and all(
            observed_process[key] == item
            for key, item in client.items()
            if key != "exe_sha256" or observed_process[key] is not None
        ),
        "Aplikacja nie potwierdza dokładnego klienta diagnostycznego",
    )
    blockers = list(PARTIAL_BLOCKERS)
    if observed_process["exe_sha256"] is None:
        blockers.append("SERVER_CLIENT_EXECUTABLE_OBSERVATION_UNAVAILABLE")
    if response["postgresql"] is None:
        blockers.append("VERIFIER_POSTGRESQL_OBSERVER_UNAVAILABLE")
    else:
        from deployment.linux_physical_postgresql import validate_service_connection_observation

        try:
            validate_service_connection_observation(
                response["postgresql"],
                challenge_id=challenge_id,
                installation_id=installation_id,
                role="verifier",
                expected_process=server,
                expected_os_principal="os_freshness_crypto_verifier",
                expected_database_role="freshness_crypto_verifier",
            )
        except ValueError as exc:
            raise LinuxIPCObservationError("Podmieniona obserwacja połączenia verifiera") from exc
    _check(response["blockers"] == blockers, "Nieuprawnione usunięcie blockerów produktu")


@contextmanager
def _pinned_verifier_socket(path: str, expected: ExpectedPeerProcess):
    """Zachowaj uchwyty chronionej ścieżki aż do końca wymiany aplikacyjnej."""
    descriptors: list[int] = []
    records = []
    try:
        parts = Path(path).parts
        _check(
            type(path) is str
            and Path(path).is_absolute()
            and str(Path(path)) == path
            and ".." not in parts
            and len(parts) >= 3,
            "Wymagany dokładny socket verifiera",
        )
        directory = os.open("/", os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(directory)
        for index, component in enumerate(parts[1:-1]):
            info = os.fstat(directory)
            _check(
                info.st_uid == 0 and not info.st_mode & 0o022,
                "Niezabezpieczona ścieżka socketu verifiera",
            )
            directory = os.open(
                component, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            descriptors.append(directory)
            records.append((directory, os.fstat(directory)))
        parent = os.fstat(directory)
        _check(
            parent.st_uid == expected.uid
            and parent.st_gid == expected.gid
            and not parent.st_mode & 0o022,
            "Nieprawidłowy katalog verifiera",
        )
        descriptor = os.open(parts[-1], os.O_PATH | os.O_NOFOLLOW, dir_fd=directory)
        descriptors.append(descriptor)
        original = os.fstat(descriptor)
        _check(
            stat.S_ISSOCK(original.st_mode)
            and original.st_uid == expected.uid
            and original.st_gid == expected.gid
            and stat.S_IMODE(original.st_mode) == 0o660
            and original.st_nlink == 1,
            "Nieprawidłowy socket rzeczywistego verifiera",
        )

        def metadata(info):
            return (
                info.st_dev,
                info.st_ino,
                info.st_uid,
                info.st_gid,
                info.st_mode,
                info.st_ctime_ns,
            )

        def recheck():
            _check(
                metadata(os.stat(parts[-1], dir_fd=directory, follow_symlinks=False))
                == metadata(original)
                == metadata(os.fstat(descriptor)),
                "Socket verifiera zmienił się podczas wymiany",
            )
            for item, before in records:
                _check(
                    metadata(os.fstat(item)) == metadata(before), "Katalog verifiera zmienił się"
                )
            for item in descriptors:
                _check(
                    not {"system.posix_acl_access", "system.posix_acl_default"}.intersection(
                        os.listxattr(f"/proc/self/fd/{item}")
                    ),
                    "Dodatkowe ACL socketu verifiera są niedozwolone",
                )

        recheck()
        yield f"/proc/self/fd/{directory}/{parts[-1]}", recheck
        recheck()
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def collect_verifier_observation(
    socket_path: str,
    *,
    expected: ExpectedPeerProcess,
    expected_cgroup: str,
    challenge_id: str,
    installation_id: str,
) -> dict[str, Any]:
    """Rzeczywisty, root-only exchange; nigdy nie udaje klienta runtime."""
    from bot_core.linux_service_ipc import (
        PinnedReadOnlyPeer,
        READONLY_OPERATION,
        capture_process_identity,
    )
    from bot_core.production_signer_ipc import (
        EXCHANGE_TIMEOUT_SECONDS,
        ProductionSignerIPCError,
        _expect_eof,
        _receive_frame,
        _remaining,
        _send_frame,
    )

    if sys.platform != "linux":
        raise LinuxIPCObservationError("Obserwacja wymaga natywnego Linux")
    _check(
        os.getresuid() == (0, 0, 0) and os.getresgid() == (0, 0, 0),
        "Odczytowa gałąź verifiera wymaga rzeczywistego root",
    )
    request = {
        "version": 1,
        "operation": READONLY_OPERATION,
        "installation_id": installation_id,
        "challenge_id": challenge_id,
        "nonce": os.urandom(32).hex(),
    }
    _check(
        all(
            type(item) is str and re.fullmatch(r"[0-9a-f]{64}", item) is not None
            for item in (challenge_id, installation_id)
        ),
        "Nieprawidłowy kontekst częściowej obserwacji",
    )
    deadline = time.monotonic() + EXCHANGE_TIMEOUT_SECONDS
    try:
        client_process = capture_process_identity(os.getpid())
        with _pinned_verifier_socket(socket_path, expected) as (address, recheck):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
                _remaining(peer, deadline)
                peer.connect(address)
                with PinnedReadOnlyPeer(peer, uid=expected.uid, gid=expected.gid) as server:
                    _expected_process(server.process, expected, expected_cgroup)
                    recheck()
                    _send_frame(peer, request, deadline)
                    peer.shutdown(socket.SHUT_WR)
                    response = _receive_frame(peer, deadline)
                    _expect_eof(peer, deadline)
                    server.recheck()
                    recheck()
                    _check(
                        capture_process_identity(os.getpid()) == client_process,
                        "Klient zmienił się podczas obserwacji",
                    )
                    result = {
                        "contract": IPC_CONTRACT,
                        "status": "PARTIAL_VERIFIER_OBSERVATION",
                        "production_qualification": "BLOCKED",
                        "request": request,
                        "response": response,
                        "client_process": client_process,
                        "server_peer_credentials": server.credentials,
                        "server_process": server.process,
                        "socket_bound_pidfd": True,
                    }
                    validate_verifier_observation(
                        result,
                        expected=expected,
                        expected_cgroup=expected_cgroup,
                        challenge_id=challenge_id,
                        installation_id=installation_id,
                    )
                    return result
    except (ProductionSignerIPCError, OSError, ValueError, KeyError, IndexError) as exc:
        raise LinuxIPCObservationError("Nie można zebrać odczytowego dowodu verifiera") from exc
