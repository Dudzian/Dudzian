"""Tests for the pytest-independent native frozen Production Trust proof."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

from deployment.windows_stage9_frozen_proof import (
    enrollment_markers,
    prove_installed_frozen_production_trust,
)


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows frozen proof")
def test_installed_frozen_production_trust_lifecycle() -> None:
    source_value = os.environ.get("CRYPTOHUNTER_STAGE9_FINAL_PACKAGE")
    if not source_value:
        pytest.skip("operator public package is supplied only to production qualification")
    prove_installed_frozen_production_trust(
        Path(source_value),
        Path(os.environ["ProgramFiles"]) / "CryptoHunter",
        Path(os.environ["ProgramData"]) / "CryptoHunter",
    )


def test_no_enrollment_marker_paths_are_canonical() -> None:
    program_data = Path("C:/ProgramData/CryptoHunter")
    assert enrollment_markers(program_data) == (
        program_data / "State/corehost.sqlite",
        program_data / "State/enrollment-finalization.json",
        program_data / "Runtime/backend-readiness.json",
        program_data / "Runtime/Verifier/verifier-readiness.json",
    )
