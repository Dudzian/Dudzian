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
    "field,value",
    [
        ("msi", None),
        ("msi", []),
        ("production_executables", "deklaracja operatora"),
        ("production_executables", False),
        ("product_version", None),
        ("product_version", True),
        ("product_version", " \t"),
        ("msi", {"sha256": True}),
        ("msi", {"sha256": "a" * 64 + "\n"}),
        ("production_executables", {"CryptoHunterBackend.exe": None}),
    ],
)
def test_malformed_metadata_has_an_explicit_refusal(
    tmp_path: Path, field: str, value: object
) -> None:
    manifest, receipt, backend = provenance_files(tmp_path)
    document = json.loads(manifest.read_text())
    document[field] = value
    manifest.write_text(json.dumps(document))
    with pytest.raises(RuntimeProvenanceError, match="malformed"):
        qualify((manifest, receipt, backend))


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


def test_live_entrypoint_refuses_before_runtime_sampling_or_soak() -> None:
    probe = Path("deployment/windows_stage10_live.ps1").read_text(encoding="utf-8")
    assert '$diagnosticArguments = @("-m", "deployment.windows_stage10_proof_validation")' in probe
    assert "python @diagnosticArguments" in probe
    assert '@("--approved-output-directory", $ApprovedDiagnosticDirectory)' in probe
    assert '@("--protected-input", $protectedInput)' in probe
    assert "@($State, $Output, $QualifiedManifest, $CleanInstallReceipt)" in probe
    assert '@("--output", $DiagnosticOutput)' in probe
    assert 'throw "DiagnosticOutput wymaga ApprovedDiagnosticDirectory"' in probe
    assert "$reportPath" not in probe
    assert "--output $State" not in probe
    assert "--output $Output" not in probe
    assert "$LASTEXITCODE -ne 0" in probe
    assert probe.rstrip().endswith('throw "WINDOWS_STAGE10_NATIVE_PROOF_AUTHORITY_UNAVAILABLE"')
    for unsupported_execution in (
        "Service-Snapshot",
        "Get-CimInstance",
        "backend-readiness.json",
        "Start-Sleep",
        "Stopwatch",
        "windows_stage10_runtime_provenance",
    ):
        assert unsupported_execution not in probe


def test_live_entrypoint_requires_separate_output_directory_authorization() -> None:
    probe = Path("deployment/windows_stage10_live.ps1").read_text(encoding="utf-8")
    missing_authorization = probe.index(
        'throw "DiagnosticOutput wymaga ApprovedDiagnosticDirectory"'
    )
    default_arguments = probe.index(
        '$diagnosticArguments = @("-m", "deployment.windows_stage10_proof_validation")'
    )
    append_directory = probe.index(
        '$diagnosticArguments += @("--approved-output-directory", $ApprovedDiagnosticDirectory)'
    )
    append_protected_input = probe.index(
        '$diagnosticArguments += @("--protected-input", $protectedInput)'
    )
    append_output = probe.index('$diagnosticArguments += @("--output", $DiagnosticOutput)')
    helper_execution = probe.index("python @diagnosticArguments")
    assert missing_authorization < default_arguments < append_directory
    assert append_directory < append_protected_input < append_output
    assert append_output < helper_execution
    for legacy_parameter in ("State", "Output"):
        assert f"${legacy_parameter}\n" not in probe
        assert f"--output ${legacy_parameter}" not in probe
    assert "@($State, $Output, $QualifiedManifest, $CleanInstallReceipt)" in probe


def test_point_identity_requires_exact_creation_time_and_program_identity() -> None:
    helper = Path("deployment/windows_stage10_soak_identity.ps1").read_text(encoding="utf-8")
    for field in ("pid", "started", "backend_path", "backend_sha256"):
        assert f"$snapshot.{field}" in helper
        assert f"$Observed.{field}" in helper
        assert f"$Expected.{field}" in helper
    assert "TryParseExact" in helper
    assert "InvariantCulture" in helper
    assert "$snapshot.pid -isnot [int]" in helper
    regression = Path("deployment/windows_stage10_soak_identity_regression.ps1").read_text(
        encoding="utf-8"
    )
    assert "Parser]::ParseFile" in regression
    assert '"windows_stage10_live.ps1"' in regression
    assert "Assert-Refused -Baseline $observed -Sample $observed" in regression
