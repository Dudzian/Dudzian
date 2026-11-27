"""Verify the measurement-only bootstrap without starting a privileged view."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
LINUX_X86_64 = sys.platform == "linux" and platform.machine() == "x86_64"
pytestmark = pytest.mark.skipif(not LINUX_X86_64, reason="Linux x86-64 bootstrap only")


@pytest.fixture(scope="module")
def compiled_probe(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[..., subprocess.CompletedProcess[bytes]]:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C toolchain unavailable")
    scratch = tmp_path_factory.mktemp("stage9-v5d11-bootstrap")
    probe = scratch / "probe.c"
    probe.write_text(
        """
#define main bootstrap_main
#include "stage9_v5d11_bootstrap.c"
#undef main

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "gate") == 0) {
        parked_gate();
        return 126;
    }
    if (argc == 3 && strcmp(argv[1], "path") == 0)
        return absolute_path(argv[2]) ? 0 : 125;
    if (argc == 3 && strcmp(argv[1], "fds") == 0) {
        int fd = atoi(argv[2]);
        if (fd <= 2 || fcntl(fd, F_GETFD) < 0)
            return 126;
        isolate_descriptors();
        return fcntl(fd, F_GETFD) < 0 && errno == EBADF ? 0 : 126;
    }
    return bootstrap_main(argc, argv);
}
""",
        encoding="ascii",
    )
    executable = scratch / "probe"
    subprocess.run(
        [
            compiler,
            "-std=c17",
            "-O2",
            "-D_FORTIFY_SOURCE=2",
            "-DSTAGE9_V5D11_ENABLE_AUTHORIZED_PREFLIGHT",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wl,--wrap=execve",
            "-Wl,--wrap=execveat",
            "-I",
            str(SCRIPTS),
            "-o",
            str(executable),
            str(probe),
            str(SCRIPTS / "stage9_v5d11_policy.c"),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )

    def run(
        *args: str, raw: bytes = b"", pass_fds: tuple[int, ...] = ()
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [str(executable), *args],
            input=raw,
            capture_output=True,
            pass_fds=pass_fds,
            timeout=5,
            check=False,
        )

    return run


@pytest.mark.parametrize("raw", [b"", b"CANCEL\n"])
def test_eof_and_exact_cancel_are_safe(compiled_probe, raw):
    result = compiled_probe("gate", raw=raw)
    assert (result.returncode, result.stdout, result.stderr) == (0, b"", b"")


def test_even_well_formed_go_cannot_dispatch(compiled_probe):
    result = compiled_probe("gate", raw=b"GO " + b"f" * 64 + b"\n")
    assert result.returncode == 125
    assert result.stderr == b"BOOTSTRAP_NATIVE_RELEASE_DISABLED\n"
    assert result.stdout == b""


@pytest.mark.parametrize("raw", [b"CANCEL\n\0", b"GO " + b"G" * 64 + b"\n", b"GO\n", b"x" * 69])
def test_gate_refuses_malformed_frames(compiled_probe, raw):
    result = compiled_probe("gate", raw=raw)
    assert result.returncode == 125
    assert result.stderr in (b"BOOTSTRAP_GATE_PROTOCOL\n", b"BOOTSTRAP_EVIDENCE_LIMIT\n")
    assert result.stdout == b""


@pytest.mark.parametrize("path", ["/safe-private/profile-a", "/safe/.name_1"])
def test_canonical_bootstrap_paths(compiled_probe, path):
    assert compiled_probe("path", path).returncode == 0


@pytest.mark.parametrize(
    "path", ["/", "relative", "/safe//file", "/safe/../unsound", "/safe/*", "/safe/hello world"]
)
def test_ambiguous_bootstrap_paths_are_refused(compiled_probe, path):
    assert compiled_probe("path", path).returncode == 125


def test_seccomp_dump_is_deterministic_and_has_classic_bpf_layout(compiled_probe):
    first = compiled_probe("--dump-seccomp")
    second = compiled_probe("--dump-seccomp")
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""
    assert 0 < len(first.stdout) < 4096 * 8
    assert len(first.stdout) % 8 == 0


def test_inherited_descriptor_is_closed_before_view_setup(compiled_probe, tmp_path):
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        result = compiled_probe("fds", str(fd), pass_fds=(fd,))
    finally:
        os.close(fd)
    assert (result.returncode, result.stdout, result.stderr) == (0, b"", b"")
