"""Rzeczywisty filtr w potomku; bez AppArmor, listenera i kwalifikacji Ubuntu."""

from __future__ import annotations

import errno
import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or platform.machine() != "x86_64",
    reason="Probe wymaga natywnego Linux x86-64 LP64 i statycznego kompilatora C",
)

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "diagnostics"
    / "stage9_hybrid_seccomp_probe.c"
)
SYSCALLS = {
    "execve",
    "execveat",
    "fork",
    "vfork",
    "clone",
    "clone3",
    "sendmsg",
    "sendmmsg",
    "recvmsg",
    "recvmmsg",
    "io_uring_setup",
    "io_uring_enter",
    "io_uring_register",
    "ptrace",
    "process_vm_readv",
    "process_vm_writev",
    "pidfd_getfd",
    "openat",
    "prctl",
    "seccomp",
    "socket",
    "bind",
    "listen",
    "accept4",
    "mmap",
    "mprotect",
    "write",
}


@pytest.fixture(scope="module")
def binaries(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("Brak kompilatora cc; próba jądrowa NOT_RUN")
    assert compiler is not None
    root = tmp_path_factory.mktemp("stage9-hybrid-native")
    builds = {
        "eacces": [],
        "eusers": ["-DPROBE_DENY_ERRNO=EUSERS"],
        "bad-flags": ["-DPROBE_FILTER_FLAGS=0x80000001U"],
        "x32-errno": ["-DPROBE_X32_ACTION=(SECCOMP_RET_ERRNO|EUSERS)"],
    }
    result = {}
    for name, extra in builds.items():
        executable = root / name
        subprocess.run(
            [
                compiler,
                "-static",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                *extra,
                str(SOURCE),
                "-o",
                str(executable),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        result[name] = executable
    return result


def run_probe(executable: Path, argument: str = "--probe") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), argument], capture_output=True, text=True, timeout=10, check=False
    )


def test_kernel_denials_change_with_only_the_filter_errno(binaries: dict[str, Path]) -> None:
    """Różnicowa odmowa odróżnia nasz filtr od odziedziczonego sandboxu/LSM."""
    for name, expected in [("eacces", errno.EACCES), ("eusers", errno.EUSERS)]:
        completed = run_probe(binaries[name])
        assert completed.returncode == 0, (completed.stdout, completed.stderr)
        assert completed.stderr == ""
        rows = [
            line.split(":") for line in completed.stdout.splitlines() if line.startswith("ODMOWA:")
        ]
        assert len(rows) == len(SYSCALLS)
        assert {row[1] for row in rows} == SYSCALLS
        assert {int(row[2]) for row in rows} == {expected}
        assert completed.stdout.endswith(
            "TEST_ONLY: lokalny filtr sprawdzony; AppArmor/Python/AF_UNIX NOT_PROVEN\n"
        )


def test_x32_guard_is_attributed_by_changing_only_its_action(binaries: dict[str, Path]) -> None:
    completed = run_probe(binaries["eacces"], "--probe-x32")
    assert completed.returncode == -signal.Signals["SIGSYS"].value
    assert completed.stdout == ""
    changed = run_probe(binaries["x32-errno"], "--probe-x32")
    assert changed.returncode == 0
    assert changed.stdout == f"ODMOWA:x32:{errno.EUSERS}\n"


def test_installation_error_blocks_before_probes(binaries: dict[str, Path]) -> None:
    completed = run_probe(binaries["bad-flags"])
    assert completed.returncode == 2
    assert completed.stdout == "BLOCKED: ustanowienie ochrony nieudane; brak listenera\n"
    assert "ODMOWA:" not in completed.stdout


def test_unknown_argument_blocks_before_initialization(binaries: dict[str, Path]) -> None:
    completed = run_probe(binaries["eacces"], "--serve")
    assert completed.returncode == 64
    assert completed.stdout.startswith("BLOCKED:")
    assert "ODMOWA:" not in completed.stdout
