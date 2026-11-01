"""Fakty jądrowe IPC nie zastępują funkcjonalnej kwalifikacji produktu.

Istniejący transport signera obsługuje DIAGNOSTIC/QUALIFY i celowo zwraca
BLOCKED. Brakuje bezpiecznego adaptera pozwalającego wykonać na rzeczywistych
usługach pełny test autoryzowanego klienta i odmowy obcej tożsamości. Ten moduł
nie otwiera połączeń z endpointami produktu ani nie wysyła arbitralnych ramek.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import socket
import struct
import sys
from typing import Any

IPC_CONTRACT = "cryptohunter-linux-ipc-security-v1"
IPC_BLOCKERS = (
    "PRODUCT_IPC_AUTHENTICATED_READONLY_ADAPTER_UNAVAILABLE",
    "AUTHORIZED_CLIENT_APPLICATION_EXCHANGE_UNPROVEN",
    "UNAUTHORIZED_CLIENT_EFFECTIVE_DENIAL_UNPROVEN",
    "SERVER_OBSERVED_CLIENT_IDENTITY_UNPROVEN",
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
