"""TEST_ONLY: dokładny diff migracji działa wyłącznie na prywatnej kopii.

BASELINE_SOURCE_SHA256 oznacza bajty main po #3257, nie zatwierdzenie wydania.
Istniejący pin pełnego AccountGenesis różni się od tej bazy i pozostaje bez zmian.
"""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

SOURCE = "bot_core/account_genesis_freshness_authority.py"
BASELINE_SOURCE_SHA256 = "81cd0df375fe1a48ba943c7645376b4d97350288c0ebe6523207ff90ed22572b"
ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "tests/fixtures/account_genesis_verified_transport_review.patch"


def load_account_genesis(tmp_path: Path) -> ModuleType:
    """Sprawdź bazę testu i załaduj proponowane bajty bez podmiany produkcji."""
    source = ROOT / SOURCE
    baseline = source.read_bytes()
    if hashlib.sha256(baseline).hexdigest() != BASELINE_SOURCE_SHA256:
        raise ValueError("Odmowa: zmieniła się dokładna baza testowej propozycji AccountGenesis")
    target = tmp_path / SOURCE
    target.parent.mkdir(parents=True)
    target.write_bytes(baseline)
    for options in (("--check",), ()):
        subprocess.run(
            ["git", "apply", "--unidiff-zero", *options, str(PATCH)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )
    name = f"bot_core._account_genesis_transport_review_{id(tmp_path)}"
    spec = importlib.util.spec_from_file_location(name, target)
    if spec is None or spec.loader is None:
        raise RuntimeError("Nie można załadować prywatnej kopii propozycji AccountGenesis")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
