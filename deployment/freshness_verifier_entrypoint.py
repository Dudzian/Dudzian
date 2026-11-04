"""Stały punkt wejścia verifiera dla izolowanego interpretera systemd.

Python -I usuwa katalog roboczy z sys.path. Tworzymy wyłącznie przestrzeń
pakietu o stałym źródle, bez skutków ubocznych bot_core/__init__.py.
"""

from __future__ import annotations

import importlib
import hashlib
import os
import runpy
import stat
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

INSTALLATION = Path("/usr/lib/cryptohunter")


def _check_installation() -> None:
    directories = (
        *reversed(INSTALLATION.parents),
        INSTALLATION,
        INSTALLATION / "deployment",
        INSTALLATION / "bot_core",
    )
    for path in directories:
        status = path.lstat()
        if not stat.S_ISDIR(status.st_mode) or status.st_uid != 0 or status.st_mode & 0o022:
            raise PermissionError("Niechroniona ścieżka kodu verifiera")
    if Path(os.path.abspath(__file__)) != INSTALLATION / "deployment" / Path(__file__).name:
        raise PermissionError("Niezgodna ścieżka punktu wejścia verifiera")


def _verify_application() -> None:
    """Odtwórz podpis i wszystkie bajty przed pierwszym importem aplikacji."""
    helper = INSTALLATION / "deployment/freshness_application.py"
    status = helper.lstat()
    if (
        not stat.S_ISREG(status.st_mode)
        or status.st_uid != 0
        or status.st_nlink != 1
        or status.st_mode & 0o022
    ):
        raise PermissionError("Niechroniony bootstrap aplikacji")
    namespace = runpy.run_path(str(helper), run_name="_cryptohunter_application_trust")
    verified = namespace["verify_installed_application"]()
    runtime = verified["runtime"]
    executable = Path("/proc/self/exe")
    approved = Path(runtime["base_executable"])
    running_status = executable.stat()
    approved_status = approved.stat()
    if (
        sys.executable != runtime["executable"]
        or (running_status.st_dev, running_status.st_ino)
        != (approved_status.st_dev, approved_status.st_ino)
        or hashlib.sha256(executable.read_bytes()).hexdigest() != runtime["base_executable_sha256"]
    ):
        raise PermissionError("Uruchomiony interpreter nie odpowiada podpisanemu runtime")


def main() -> int:
    _check_installation()
    # -B blokuje zapis, ale nadal czyta zastany bytecode. Nie używamy cache
    # lokalnego payloadu; /dev/null nie może zawierać podkatalogów cache.
    sys.pycache_prefix = "/dev/null"
    if any(name == "bot_core" or name.startswith("bot_core.") for name in sys.modules):
        raise PermissionError("Pakiet verifiera został załadowany przed punktem wejścia")
    _verify_application()
    # Interpreter, jego pakiety i root pozostają granicą zaufania. Dokładne
    # bajty payloadu zostały ponownie zweryfikowane przed importem usługi.
    package = ModuleType("bot_core")
    package.__package__ = "bot_core"
    package.__path__ = [str(INSTALLATION / "bot_core")]
    package.__spec__ = ModuleSpec("bot_core", loader=None, is_package=True)
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules["bot_core"] = package
    service = importlib.import_module("bot_core.freshness_verifier_service")
    service.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
