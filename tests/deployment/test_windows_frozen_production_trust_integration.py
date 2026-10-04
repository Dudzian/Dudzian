"""Opt-in native proof of the installed, one-file Production Trust command.

This test intentionally cannot fall back to repository Python.  The public final
package is operator supplied because completed production ceremony artifacts do
not belong in the source tree.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from deployment.windows_stage9_production_trust import CEREMONY_ID, load_production_trust


def _hashes(path: Path) -> dict[str, str]:
    return {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.iterdir())
        if item.is_file()
    }


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows frozen proof")
def test_installed_frozen_production_trust_lifecycle() -> None:
    source_value = os.environ.get("CRYPTOHUNTER_STAGE9_FINAL_PACKAGE")
    if not source_value:
        pytest.skip("operator public package not supplied")

    source = Path(source_value)
    program_files = Path(os.environ["ProgramFiles"]) / "CryptoHunter"
    program_data = Path(os.environ["ProgramData"]) / "CryptoHunter"
    executable = program_files / "CryptoHunterProvision.exe"
    destination = program_data / "Config" / "ProductionTrust" / CEREMONY_ID
    markers = (
        program_data / "State" / "corehost.sqlite",
        program_data / "enrollment-finalization.json",
        program_data / "Runtime" / "backend-readiness.json",
        program_data / "Runtime" / "Verifier" / "verifier-readiness.json",
    )
    assert executable.is_file()
    assert not destination.exists()
    assert not any(path.exists() for path in markers)
    load_production_trust(source)
    print("FROZEN_PRODUCTION_TRUST_SOURCE_VERIFY = PASS")

    try:
        subprocess.run(
            [
                str(executable),
                "install-production-trust",
                "--program-files",
                str(program_files),
                "--program-data",
                str(program_data),
                "--source",
                str(source),
            ],
            check=True,
            timeout=120,
        )
        print("FROZEN_PRODUCTION_TRUST_INSTALL = PASS")
        assert _hashes(destination) == _hashes(source)
        print("FROZEN_PRODUCTION_TRUST_BYTE_IDENTITY = PASS")
        subprocess.run(
            [
                str(executable),
                "qualify-dacl",
                "--program-files",
                str(program_files),
                "--program-data",
                str(program_data),
            ],
            check=True,
            timeout=120,
        )
        print("FROZEN_PRODUCTION_TRUST_DACL = PASS")
        assert not any(path.exists() for path in markers)
        print("FROZEN_PRODUCTION_TRUST_NO_ENROLLMENT = PASS")
    finally:
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        parent = destination.parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    assert not destination.exists()
    assert not tuple((program_data / "Config").glob("ProductionTrust/.production-trust-*"))
