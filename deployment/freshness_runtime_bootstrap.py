"""Minimalna granica zaufania runtime: stdlib i chroniony OpenSSL systemu.

Moduł nigdy nie importuje bibliotek ani nie uruchamia interpretera z bundle.
Klucz pochodzi z odrębnej ceremonii administratora, nie z samego artefaktu.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path


class RuntimeTrustError(RuntimeError):
    """Materiały nie ustanawiają niezależnej granicy zaufania."""


def _openssl() -> str:
    executable = Path("/usr/bin/openssl")
    for item in (*reversed(executable.parents), executable):
        status = item.lstat()
        if stat.S_ISLNK(status.st_mode) or status.st_uid != 0 or status.st_mode & 0o022:
            raise RuntimeTrustError("Niechroniony bootstrap OpenSSL")
    if not executable.is_file():
        raise RuntimeTrustError("Brak systemowego bootstrap OpenSSL")
    return str(executable)


def verify_ed25519(raw: bytes, signature: bytes, key_bytes: bytes) -> None:
    """Sprawdź Ed25519 bez importu cryptography i bez zapisu na dysk.

    Anonimowe deskryptory memfd zapewniają interfejs plikowy OpenSSL. Nie ma
    ścieżek tymczasowych, plików hosta ani wykonania danych przed weryfikacją.
    """
    if sys.platform != "linux":
        raise RuntimeTrustError("Bootstrap runtime wymaga natywnego Linux")
    if (
        type(raw) is not bytes
        or not 0 < len(raw) <= 32 * 1024 * 1024
        or type(signature) is not bytes
        or len(signature) != 64
        or type(key_bytes) is not bytes
        or len(key_bytes) != 32
        or not hasattr(os, "memfd_create")
    ):
        raise RuntimeTrustError("Nieprawidłowe materiały Ed25519 lub brak memfd")
    descriptors: list[int] = []
    try:
        public_der = bytes.fromhex("302a300506032b6570032100") + key_bytes
        for value in (raw, signature, public_der):
            fd = os.memfd_create("cryptohunter-ed25519", os.MFD_CLOEXEC)
            descriptors.append(fd)
            cursor = 0
            while cursor < len(value):
                cursor += os.write(fd, value[cursor:])
            os.lseek(fd, 0, os.SEEK_SET)
        result = subprocess.run(
            [
                _openssl(),
                "pkeyutl",
                "-verify",
                "-rawin",
                "-pubin",
                "-keyform",
                "DER",
                "-inkey",
                f"/proc/self/fd/{descriptors[2]}",
                "-sigfile",
                f"/proc/self/fd/{descriptors[1]}",
                "-in",
                f"/proc/self/fd/{descriptors[0]}",
            ],
            pass_fds=tuple(descriptors),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            cwd="/",
            timeout=15,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeTrustError("Nieprawidłowy podpis niezależnego autorytetu")
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeTrustError("Bootstrap weryfikacji podpisu niedostępny") from exc
    finally:
        for fd in descriptors:
            os.close(fd)


def main() -> int:
    """ExecStartPre: sprawdź chroniony core przed odczytem runtime."""
    import runpy
    import sys

    if sys.argv[1:] != ["verify-installed"]:
        print("Odmowa: nieprawidłowa operacja bootstrap runtime")
        return 1
    try:
        core = Path(__file__).absolute().with_name("freshness_runtime.py")
        for component in (*reversed(core.parents), core):
            info = component.lstat()
            if info.st_uid != 0 or info.st_mode & 0o022 or stat.S_ISLNK(info.st_mode):
                raise RuntimeTrustError("Niechroniony core runtime")
        sys.pycache_prefix = "/dev/null"
        namespace = runpy.run_path(str(core), run_name="_cryptohunter_runtime_core")
        namespace["verify_installed_runtime"]()
    except (RuntimeError, OSError, ValueError):
        print("Odmowa: granica zaufania runtime nie została ustanowiona")
        return 1
    print("Integralność podpisanego runtime zweryfikowana")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
