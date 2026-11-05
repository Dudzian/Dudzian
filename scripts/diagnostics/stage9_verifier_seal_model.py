"""TEST_ONLY: model sekwencji i kompilator cBPF, bez wykonywania syscalli.

Wynik MODELED nigdy nie kwalifikuje launchera, import closure ani tożsamości
żywego verifiera. Ten moduł nie zakłada filtra i nie otwiera listenera.
Numery syscalli dotyczą wyłącznie Linux x86_64/LP64 (asm/unistd_64.h).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Literal

AUDIT_ARCH_X86_64 = 0xC000003E
X32_BIT = 0x40000000
ALLOW = 0x7FFF0000
DENY = 0x00050001  # SECCOMP_RET_ERRNO | EPERM; tylko wynik ewaluatora.
KILL = 0x80000000  # SECCOMP_RET_KILL_PROCESS; niczego tutaj nie zabija.
LD, JEQ, JSET, RET = 0x20, 0x15, 0x45, 0x06

# Brak open*, exec*, clone*, prctl/seccomp, credentials, namespaces, ioctl,
# ptrace/pidfd_getfd, sendmsg/sendmmsg/recvmsg/recvmmsg i całego io_uring.
# Jest to kandydat do badań, nie zakwalifikowany profil CPython/Freshness.
SIMPLE_CALLS = (
    0,
    1,
    3,
    5,
    7,
    8,
    11,
    12,
    13,
    14,
    15,
    28,
    39,
    44,
    45,
    48,
    49,
    50,
    60,
    102,
    104,
    107,
    108,
    202,
    228,
    231,
    271,
    318,
)
SYSCALLS = {
    "read": 0,
    "write": 1,
    "mmap": 9,
    "mprotect": 10,
    "socket": 41,
    "fork": 57,
    "vfork": 58,
    "clone": 56,
    "execve": 59,
    "sendmsg": 46,
    "recvmsg": 47,
    "ptrace": 101,
    "prctl": 157,
    "accept4": 288,
    "recvmmsg": 299,
    "sendmmsg": 307,
    "seccomp": 317,
    "execveat": 322,
    "io_uring_setup": 425,
    "io_uring_enter": 426,
    "io_uring_register": 427,
    "clone3": 435,
    "pidfd_getfd": 438,
}


@dataclass(frozen=True)
class Instruction:
    code: int
    jt: int = 0
    jf: int = 0
    k: int = 0


def _equal(offset: int, value: int) -> list[Instruction]:
    return [Instruction(LD, k=offset), Instruction(JEQ, jt=1, k=value), Instruction(RET, k=DENY)]


def _argument(index: int, value: int) -> list[Instruction]:
    """Pełne porównanie argumentu 64-bit; bez pominięcia górnego słowa."""
    offset = 16 + index * 8
    return _equal(offset + 4, value >> 32) + _equal(offset, value & 0xFFFFFFFF)


def _no_exec(index: int) -> list[Instruction]:
    offset = 16 + index * 8
    return _equal(offset + 4, 0) + [
        Instruction(LD, k=offset),
        Instruction(JSET, jf=1, k=4),
        Instruction(RET, k=DENY),
    ]


def compile_filter() -> tuple[Instruction, ...]:
    """Zbuduj konkretne instrukcje sock_filter; nie instaluj ich w jądrze.

    mmap dopuszcza tylko anonimowe MAP_PRIVATE, fd=-1 i offset=0; mprotect
    nie może dodać PROT_EXEC. socket wyłącznie AF_UNIX/STREAM/CLOEXEC.
    Filtr nie potrafi sprawdzić ścieżki bind, treści wskaźnika ani kodu Python.
    """
    program = [
        Instruction(LD, k=4),
        Instruction(JEQ, jt=1, k=AUDIT_ARCH_X86_64),
        Instruction(RET, k=KILL),
        Instruction(LD, k=0),
        Instruction(JSET, jf=1, k=X32_BIT),
        Instruction(RET, k=KILL),
    ]
    checked = {
        9: _no_exec(2) + _argument(3, 0x22) + _argument(4, 0xFFFFFFFFFFFFFFFF) + _argument(5, 0),
        10: _no_exec(2),
        41: _argument(0, 1) + _argument(1, 1 | 0x80000) + _argument(2, 0),
        288: _argument(3, 0x80000),
    }
    for number in sorted((*SIMPLE_CALLS, *checked)):
        conditions = checked.get(number, [])
        program.extend([Instruction(LD, k=0), Instruction(JEQ, jf=len(conditions) + 1, k=number)])
        program.extend(conditions)
        program.append(Instruction(RET, k=ALLOW))
    program.append(Instruction(RET, k=DENY))
    return tuple(program)


def encode_filter(program: tuple[Instruction, ...]) -> bytes:
    """Kodowanie sock_filter dla zadeklarowanego little-endian ABI."""
    if not 0 < len(program) <= 4096:
        raise ValueError("Nieprawidłowa długość programu cBPF")
    return b"".join(struct.pack("<HBBI", op.code, op.jt, op.jf, op.k) for op in program)


def seccomp_data(
    number: int, args: tuple[int, ...] = (), *, architecture: int = AUDIT_ARCH_X86_64
) -> bytes:
    """Syntetyczne 64 bajty seccomp_data; brak odczytu stanu systemu."""
    if len(args) > 6 or any(type(value) is not int or not 0 <= value < 2**64 for value in args):
        raise ValueError("Nieprawidłowe argumenty syntetycznego syscalla")
    return struct.pack("<iIQ6Q", number, architecture, 0, *(args + (0,) * (6 - len(args))))


def evaluate(program: tuple[Instruction, ...], data: bytes) -> int:
    """Ograniczony interpreter cBPF; nie jest Linuxowym verifierem BPF."""
    if len(data) != 64 or not 0 < len(program) <= 4096:
        raise ValueError("Nieprawidłowy model programu lub seccomp_data")
    accumulator, position = 0, 0
    while position < len(program):
        op = program[position]
        if op.code == RET:
            return op.k
        if op.code == LD:
            if op.k % 4 or not 0 <= op.k <= 60:
                raise ValueError("Nieprawidłowy odczyt cBPF")
            accumulator = struct.unpack_from("<I", data, op.k)[0]
        elif op.code in (JEQ, JSET):
            if not 0 <= op.jt <= 255 or not 0 <= op.jf <= 255:
                raise ValueError("Nieprawidłowy skok cBPF")
            matched = accumulator == op.k if op.code == JEQ else bool(accumulator & op.k)
            position += op.jt if matched else op.jf
        else:
            raise ValueError("Nieobsługiwana instrukcja cBPF")
        position += 1
    raise ValueError("Program cBPF nie zakończył się RET")


@dataclass(frozen=True)
class Decision:
    result: Literal["ACCEPT", "REJECT"]
    evidence: Literal["MODELED"] = "MODELED"
    native: Literal["NOT_RUN"] = "NOT_RUN"


def syscall_decision(
    number: int, args: tuple[int, ...] = (), *, architecture: int = AUDIT_ARCH_X86_64
) -> Decision:
    action = evaluate(compile_filter(), seccomp_data(number, args, architecture=architecture))
    return Decision("ACCEPT" if action == ALLOW else "REJECT")


STEPS = (
    "trusted_start",
    "dumpable_zero",
    "imports_closed",
    "all_fds_closed",
    "no_new_privs",
    "filter_tsync",
    "socket",
    "bind",
    "listen",
)


@dataclass(frozen=True)
class Assumptions:
    """Wyłącznie deklaracje scenariusza, bez pomiaru i bez atestacji."""

    approved_closure: bool
    thread_count: int = 1
    tracer_pid: int = 0
    inherited_listener: bool = False
    remaining_fds: tuple[int, ...] = ()


@dataclass
class SealModel:
    assumptions: Assumptions
    position: int = 0
    failed: bool = False

    def advance(self, step: str, *, succeeded: bool = True) -> Decision:
        """Błąd, zła kolejność lub niepełne założenia zatrzaskują odmowę.

        all_fds_closed obejmuje również 0, 1, 2 i FD io_uring, ale nie dowodzi
        braku wcześniejszych mapowań lub zleceń. To model końcowego zamknięcia
        FD, bez osobnego wczesnego oczyszczenia opisanego w projekcie.
        filter_tsync oznacza jedynie model wymaganego sukcesu TSYNC, nie status 2.
        """
        facts = self.assumptions
        invalid_start = step == "trusted_start" and (
            facts.approved_closure is not True
            or type(facts.thread_count) is not int
            or facts.thread_count != 1
            or type(facts.tracer_pid) is not int
            or facts.tracer_pid != 0
            or facts.inherited_listener is not False
        )
        invalid_fds = step == "all_fds_closed" and (
            type(facts.remaining_fds) is not tuple or bool(facts.remaining_fds)
        )
        if (
            self.failed
            or self.position >= len(STEPS)
            or step != STEPS[self.position]
            or succeeded is not True
            or invalid_start
            or invalid_fds
        ):
            self.failed = True
            return Decision("REJECT")
        self.position += 1
        return Decision("ACCEPT")

    def listener_decision(self) -> Decision:
        return Decision("ACCEPT" if not self.failed and self.position == len(STEPS) else "REJECT")


@dataclass(frozen=True)
class PeerSnapshot:
    """Syntetyczne, niezależne wejścia klienta; nie odpowiedź serwera."""

    peer_pid: int
    main_pid: int
    uid: int
    gid: int
    start_time: int
    unit_epoch: str
    socket_inode: tuple[int, int]
    pidfd_alive: bool = True
    protected_socket: bool = True


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _valid_peer(snapshot: object) -> bool:
    if not isinstance(snapshot, PeerSnapshot):
        return False
    return (
        all(
            _positive_int(value)
            for value in (
                snapshot.peer_pid,
                snapshot.main_pid,
                snapshot.uid,
                snapshot.gid,
                snapshot.start_time,
            )
        )
        and type(snapshot.unit_epoch) is str
        and bool(snapshot.unit_epoch)
        and type(snapshot.socket_inode) is tuple
        and len(snapshot.socket_inode) == 2
        and all(_positive_int(value) for value in snapshot.socket_inode)
        and snapshot.pidfd_alive is True
        and snapshot.protected_socket is True
    )


def exchange_decision(
    before: PeerSnapshot,
    after: PeerSnapshot,
    *,
    client_uid: int,
    expected_uid: int,
    expected_gid: int,
) -> Decision:
    """Sprawdź spójność modelu; nie uwierzytelniaj rzeczywistego endpointu."""
    valid = (
        _valid_peer(before)
        and _valid_peer(after)
        and all(_positive_int(value) for value in (client_uid, expected_uid, expected_gid))
        and before == after
        and before.peer_pid == before.main_pid
        and before.uid == expected_uid != client_uid
        and before.gid == expected_gid
    )
    return Decision("ACCEPT" if valid else "REJECT")
