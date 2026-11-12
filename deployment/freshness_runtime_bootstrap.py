"""Minimalna granica zaufania runtime: stdlib i chroniony OpenSSL systemu.

Moduł nigdy nie importuje bibliotek ani nie uruchamia interpretera z bundle.
Klucz pochodzi z odrębnej ceremonii administratora, nie z samego artefaktu.
"""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import sys
from pathlib import Path

# Kontrakt należy do niezależnie zaufanego bootstrapu, nigdy do manifestu,
# środowiska procesu ani pliku openssl.cnf hosta. Brak .include/engine/module.
OPENSSL_CONFIGURATION = (
    b"config_diagnostics = 1\n"
    b"openssl_conf = cryptohunter_bootstrap\n"
    b"[cryptohunter_bootstrap]\n"
    b"providers = cryptohunter_providers\n"
    b"[cryptohunter_providers]\n"
    b"default = cryptohunter_default\n"
    b"[cryptohunter_default]\n"
    b"activate = 1\n"
)
OPENSSL_CONFIGURATION_SHA256 = "c77df0bc24b7bfc5b935de2097109c144c0f52fd4f2542785881ea230d145bf6"
OPENSSL_PROVIDER = "default"


class RuntimeTrustError(RuntimeError):
    """Materiały nie ustanawiają niezależnej granicy zaufania."""


def _configuration() -> bytes:
    """Odrzuć zmieniony kontrakt przed uruchomieniem OpenSSL 3."""
    if (
        type(OPENSSL_CONFIGURATION) is not bytes
        or hashlib.sha256(OPENSSL_CONFIGURATION).hexdigest() != OPENSSL_CONFIGURATION_SHA256
        or OPENSSL_PROVIDER != "default"
    ):
        raise RuntimeTrustError("Niezatwierdzona konfiguracja lub provider bootstrapu")
    return OPENSSL_CONFIGURATION


def _sealed_input(value: bytes) -> int:
    if sys.platform != "linux":
        raise RuntimeTrustError("Bootstrap runtime wymaga natywnego Linux")
    import fcntl

    fd = os.memfd_create("cryptohunter-ed25519", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        cursor = 0
        while cursor < len(value):
            written = os.write(fd, value[cursor:])
            if written <= 0:
                raise RuntimeTrustError("Niepełny zapis materiału bootstrapu")
            cursor += written
        os.lseek(fd, 0, os.SEEK_SET)
        # Linux UAPI fcntl.h: część kompilacji CPython nie eksportuje nazw
        # F_*_SEALS mimo dostępnej obsługi jądra. Błąd syscall nadal odmawia.
        fcntl.fcntl(
            fd,
            getattr(fcntl, "F_ADD_SEALS", 1033),
            0x000F,  # F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE
        )
        return fd
    except BaseException:
        os.close(fd)
        raise


def _openssl() -> str:
    executable = Path("/usr/bin/openssl")
    for item in (*reversed(executable.parents), executable):
        status = item.lstat()
        last = item == executable
        if (
            status.st_uid != 0
            or status.st_mode & 0o022
            or (
                not stat.S_ISREG(status.st_mode)
                or status.st_nlink != 1
                or not status.st_mode & stat.S_IXUSR
                if last
                else not stat.S_ISDIR(status.st_mode)
            )
        ):
            raise RuntimeTrustError("Niechroniony bootstrap OpenSSL")
    return str(executable)


def verify_ed25519(raw: bytes, signature: bytes, key_bytes: bytes) -> None:
    """Sprawdź Ed25519 bez importu cryptography i bez zapisu na dysk.

    Zapieczętowane memfd zawierają dane i dokładną konfigurację bootstrapu.
    Wbudowany provider default jest jedynym dozwolonym; ścieżka providerów
    wskazuje zwykły plik memfd, więc nie pozwala otworzyć żadnego modułu .so.
    Loader, libcrypto i pozostałe biblioteki systemu wymagają osobnego dowodu
    pochodzenia; ta funkcja nie ustanawia ich integralności.
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
        configuration = _configuration()
        public_der = bytes.fromhex("302a300506032b6570032100") + key_bytes
        for value in (raw, signature, public_der, configuration):
            descriptors.append(_sealed_input(value))
        config_path = f"/proc/self/fd/{descriptors[3]}"
        result = subprocess.run(
            [
                _openssl(),
                "pkeyutl",
                "-provider-path",
                config_path,
                "-provider",
                OPENSSL_PROVIDER,
                "-propquery",
                "provider=default",
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
            env={
                "PATH": "/usr/bin:/bin",
                "LC_ALL": "C",
                "OPENSSL_CONF": config_path,
                "OPENSSL_CONF_INCLUDE": config_path,
                "OPENSSL_MODULES": config_path,
                "OPENSSL_ENGINES": config_path,
            },
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


def _protected_helper(filename: str) -> Path:
    helper = Path(__file__).absolute().with_name(filename)
    for component in (*reversed(helper.parents), helper):
        info = component.lstat()
        if (
            info.st_uid != 0
            or info.st_mode & 0o022
            or (
                not stat.S_ISDIR(info.st_mode)
                if component != helper
                else not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            )
        ):
            raise RuntimeTrustError("Niechroniony helper bootstrapu")
    return helper


def main() -> int:
    """ExecStartPre: sprawdź runtime i aplikację przed wykonaniem launchera."""
    import runpy
    import sys

    if sys.argv[1:] != ["verify-installed"]:
        print("Odmowa: nieprawidłowa operacja bootstrap runtime")
        return 1
    try:
        core = _protected_helper("freshness_runtime.py")
        application = _protected_helper("freshness_application.py")
        sys.pycache_prefix = "/dev/null"
        namespace = runpy.run_path(str(core), run_name="_cryptohunter_runtime_core")
        namespace["verify_installed_runtime"]()
        # Te trzy helpery instalator porównuje z własną, niezależnie zaufaną
        # kopią przed publikacją. Podpis payloadu nie wybiera weryfikatora.
        namespace = runpy.run_path(str(application), run_name="_cryptohunter_application_trust")
        namespace["verify_installed_application"]()
    except (RuntimeError, OSError, ValueError):
        print("Odmowa: granica zaufania runtime nie została ustanowiona")
        return 1
    print("Integralność podpisanego runtime zweryfikowana")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
