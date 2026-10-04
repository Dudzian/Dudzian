from __future__ import annotations

from pathlib import Path

import pytest

from deployment.windows_stage9_production_trust import (
    CEREMONY_ID,
    PUBLIC_PACKAGE_FILENAMES,
    ProductionTrustUnavailable,
    validate_public_package_layout,
)


def _package(path: Path) -> Path:
    path.mkdir()
    for name in PUBLIC_PACKAGE_FILENAMES:
        (path / name).write_text("{}\n", encoding="utf-8")
    return path


def test_canonical_download_directory_and_flat_twelve_json_layout_are_accepted(
    tmp_path: Path,
) -> None:
    validate_public_package_layout(_package(tmp_path / CEREMONY_ID))


def test_wrong_download_directory_basename_is_blocked(tmp_path: Path) -> None:
    with pytest.raises(ProductionTrustUnavailable, match="CEREMONY_DIRECTORY_REQUIRED"):
        validate_public_package_layout(_package(tmp_path / "stage9-production-trust"))


@pytest.mark.parametrize("mutation", ["missing", "extra", "nested"])
def test_non_exact_operator_archive_layout_is_blocked(tmp_path: Path, mutation: str) -> None:
    package = _package(tmp_path / CEREMONY_ID)
    if mutation == "missing":
        (package / next(iter(PUBLIC_PACKAGE_FILENAMES))).unlink()
    elif mutation == "extra":
        (package / "extra.json").write_text("{}\n", encoding="utf-8")
    else:
        (package / "wrapper").mkdir()
    with pytest.raises(ProductionTrustUnavailable, match="ALLOWLIST_VIOLATION"):
        validate_public_package_layout(package)
