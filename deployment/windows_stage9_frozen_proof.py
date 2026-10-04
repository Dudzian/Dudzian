"""Native, pytest-independent proof of the installed Stage-9 trust command."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from typing import Sequence

from deployment.windows_stage9_production_trust import (
    CEREMONY_ID,
    load_production_trust,
    validate_public_package_layout,
)


class FrozenProductionTrustError(RuntimeError):
    """The installed executable failed its native production-trust proof."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrozenProductionTrustError(message)


def _hashes(path: Path) -> dict[str, str]:
    return {item.name: hashlib.sha256(item.read_bytes()).hexdigest() for item in path.iterdir()}


def enrollment_markers(program_data: Path) -> tuple[Path, ...]:
    return (
        program_data / "State/corehost.sqlite",
        program_data / "State/enrollment-finalization.json",
        program_data / "Runtime/backend-readiness.json",
        program_data / "Runtime/Verifier/verifier-readiness.json",
    )


def prove_installed_frozen_production_trust(
    source: Path, program_files: Path, program_data: Path
) -> None:
    """Verify and exercise the frozen executable without a test-runner dependency."""
    validate_public_package_layout(source)
    load_production_trust(source)
    print("FROZEN_PRODUCTION_TRUST_SOURCE_VERIFY = PASS", flush=True)
    executable = program_files / "CryptoHunterProvision.exe"
    destination = program_data / "Config/ProductionTrust" / CEREMONY_ID
    markers = enrollment_markers(program_data)
    _require(executable.is_file(), "installed frozen provision executable absent")
    _require(not destination.exists(), "production trust destination already exists")
    _require(not any(path.exists() for path in markers), "enrollment marker exists before proof")
    try:
        subprocess.run(
            [str(executable), "install-production-trust", "--program-files", str(program_files),
             "--program-data", str(program_data), "--source", str(source)],
            check=True, timeout=120,
        )
        print("FROZEN_PRODUCTION_TRUST_INSTALL = PASS", flush=True)
        _require(_hashes(destination) == _hashes(source), "installed package byte identity differs")
        print("FROZEN_PRODUCTION_TRUST_BYTE_IDENTITY = PASS", flush=True)
        subprocess.run(
            [str(executable), "qualify-dacl", "--program-files", str(program_files),
             "--program-data", str(program_data)],
            check=True, timeout=120,
        )
        print("FROZEN_PRODUCTION_TRUST_DACL = PASS", flush=True)
        _require(not any(path.exists() for path in markers), "frozen proof performed enrollment")
        print("FROZEN_PRODUCTION_TRUST_NO_ENROLLMENT = PASS", flush=True)
    finally:
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        parent = destination.parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    _require(not destination.exists(), "production trust cleanup failed")
    _require(
        not tuple((program_data / "Config").glob("ProductionTrust/.production-trust-*")),
        "production trust staging residue remains",
    )
    print("FROZEN_PRODUCTION_TRUST_CLEANUP = PASS", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args(argv)
    prove_installed_frozen_production_trust(
        args.source,
        Path(os.environ["ProgramFiles"]) / "CryptoHunter",
        Path(os.environ["ProgramData"]) / "CryptoHunter",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
