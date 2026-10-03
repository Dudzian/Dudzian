from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from deployment.windows_stage10_runtime_provenance import (
    RuntimeProvenanceError,
    qualify_runtime_provenance,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    backend = tmp_path / "CryptoHunterBackend.exe"
    backend.write_bytes(b"current backend")
    manifest = tmp_path / "installer-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "product_version": "1.2.3",
                "architecture": "x64",
                "msi": {"file": "current.msi", "sha256": "a" * 64},
                "production_executables": {"CryptoHunterBackend.exe": _sha(backend)},
            }
        )
    )
    receipt = tmp_path / "clean-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "source_revision": "current",
                "ci_run_id": "42",
                "ci_provider": "https://github.com",
                "runner_os": "Windows",
                "product_version": "1.2.3",
                "manifest_sha256": _sha(manifest),
                "msi_sha256": "a" * 64,
            }
        )
    )
    return manifest, receipt, backend


def qualify(paths: tuple[Path, Path, Path]) -> dict[str, str]:
    return qualify_runtime_provenance(
        *paths,
        "1.2.3",
        source_revision="current",
        ci_run_id="42",
        ci_provider="https://github.com",
    )


def test_current_run_manifest_is_bound_to_installed_backend(tmp_path: Path) -> None:
    manifest, receipt, backend = provenance_files(tmp_path)
    result = qualify((manifest, receipt, backend))
    assert result == {
        "source_revision": "current",
        "ci_run_id": "42",
        "product_version": "1.2.3",
        "qualified_manifest_sha256": _sha(manifest),
        "qualified_msi_sha256": "a" * 64,
        "installed_backend_sha256": _sha(backend),
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda _m, receipt, _b: receipt.update(source_revision="stale"),
        lambda _m, receipt, _b: receipt.update(ci_run_id="old-run"),
        lambda _m, receipt, _b: receipt.update(manifest_sha256="b" * 64),
        lambda _m, receipt, _b: receipt.update(msi_sha256="b" * 64),
        lambda manifest, _r, _b: manifest["production_executables"].update(
            {"CryptoHunterBackend.exe": "b" * 64}
        ),
    ],
)
def test_stale_revision_run_manifest_msi_or_backend_is_rejected(
    tmp_path: Path,
    mutation,
) -> None:
    manifest_path, receipt_path, backend = provenance_files(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    mutation(manifest, receipt, backend)
    manifest_path.write_text(json.dumps(manifest))
    if receipt.get("manifest_sha256") != "b" * 64:
        receipt["manifest_sha256"] = _sha(manifest_path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeProvenanceError, match="not the current-run"):
        qualify((manifest_path, receipt_path, backend))


def test_missing_installed_backend_and_unexpected_product_version_are_rejected(
    tmp_path: Path,
) -> None:
    paths = provenance_files(tmp_path)
    paths[2].unlink()
    with pytest.raises(RuntimeProvenanceError, match="missing"):
        qualify(paths)
    paths = provenance_files(tmp_path)
    with pytest.raises(RuntimeProvenanceError, match="not the current-run"):
        qualify_runtime_provenance(
            *paths,
            "9.9.9",
            source_revision="current",
            ci_run_id="42",
            ci_provider="https://github.com",
        )


def test_post_reboot_and_soak_samples_rehash_the_qualified_executable() -> None:
    probe = Path("deployment/windows_stage10_live.ps1").read_text(encoding="utf-8")
    expected = (
        "Service-Snapshot $runtimeProvenance.installed_backend_sha256 "
        "$runtimeProvenance.product_version"
    )
    assert probe.count(expected) >= 3
    assert "$afterReboot =" + " " + expected in probe
    assert "$sample =" + " " + expected in probe
    assert "$final =" + " " + expected in probe


def test_missing_runtime_provenance_is_rejected_before_post_reboot_soak() -> None:
    probe = Path("deployment/windows_stage10_live.ps1").read_text(encoding="utf-8")
    assert "$null -eq $runtimeProvenance" in probe
    assert 'throw "missing or stale runtime provenance"' in probe
