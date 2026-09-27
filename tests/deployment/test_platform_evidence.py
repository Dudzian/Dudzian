from __future__ import annotations

import json
from pathlib import Path

import pytest

from deployment.platform_evidence import (
    EvidenceProductionError,
    aggregate_core_markers,
    produce_windows_clean_install_evidence,
)
from deployment.core_test_plan import load_manifest, marker_document


def test_core_aggregation_requires_same_revision_and_all_operating_systems(
    tmp_path: Path,
) -> None:
    markers = []
    manifest = load_manifest()
    for runner_os in ("Linux", "Windows", "macOS"):
        marker = tmp_path / f"{runner_os}.json"
        marker.write_text(
            json.dumps(
                marker_document(manifest, "current", runner_os, "run-1", "https://github.com")
            ),
            encoding="utf-8",
        )
        markers.append(marker)
    output = tmp_path / "core.json"
    aggregate_core_markers(
        "current",
        markers,
        output,
        ci_run_id="run-1",
        ci_provider="https://github.com",
    )
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["platform"] == "CROSS_PLATFORM_CORE"
    assert evidence["source_revision"] == "current"
    assert evidence["results"][0]["item"] == "CORE_REQUIRED_SUITES"

    stale = tmp_path / "Linux.json"
    stale.write_text(
        json.dumps(marker_document(manifest, "previous", "Linux", "run-1", "https://github.com")),
        encoding="utf-8",
    )
    with pytest.raises(EvidenceProductionError, match="stale"):
        aggregate_core_markers(
            "current",
            markers,
            output,
            ci_run_id="run-1",
            ci_provider="https://github.com",
        )


def test_clean_install_evidence_requires_complete_live_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("RUNNER_OS", "Windows")
    monkeypatch.setenv("RUNNER_ARCH", "X64")
    msi = tmp_path / "product.msi"
    manifest = tmp_path / "installer-manifest.json"
    msi.write_bytes(b"msi")
    import hashlib

    msi_hash = hashlib.sha256(msi.read_bytes()).hexdigest()
    manifest.write_text(
        json.dumps(
            {
                "product_version": "1.2.3",
                "architecture": "x64",
                "wix_version": "7.0.0",
                "postgresql_version": "17.11",
                "postgresql_packaging_revision": "4",
                "msi": {"sha256": msi_hash},
            }
        )
    )

    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_revision": "revision",
                "ci_provider": "https://github.com",
                "ci_run_id": "42",
                "runner_os": "Windows",
                "probe_id": "cryptohunter.windows.clean-install.v1",
                "msi_sha256": msi_hash,
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "product_version": "1.2.3",
                "runner_arch": "X64",
                "install_exit_code": 0,
                "uninstall_exit_code": 0,
                "proofs": {
                    name: "PASS"
                    for name in (
                        "files",
                        "services",
                        "dacl",
                        "postgresql",
                        "mtls_matrix",
                        "backend",
                        "logging",
                        "uninstall",
                        "acceptance_cleanup",
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "evidence.json"
    produce_windows_clean_install_evidence("revision", receipt, output, msi=msi, manifest=manifest)
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["results"][0]["item"] == "WINDOWS_CLEAN_INSTALL"
    assert evidence["results"][0]["status"] == "PASS"
    receipt_value = json.loads(receipt.read_text())
    receipt_value["proofs"]["mtls_matrix"] = "FAIL"
    receipt.write_text(json.dumps(receipt_value))
    with pytest.raises(EvidenceProductionError, match="incomplete or failed"):
        produce_windows_clean_install_evidence(
            "revision", receipt, output, msi=msi, manifest=manifest
        )


def test_core_aggregation_rejects_stale_plan_duplicate_runner_and_wrong_run(
    tmp_path: Path,
) -> None:
    manifest = load_manifest()

    def write(name: str, runner_os: str, **updates: str) -> Path:
        value = marker_document(manifest, "current", runner_os, "run-1", "https://github.com")
        value.update(updates)
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    linux = write("linux.json", "Linux")
    windows = write("windows.json", "Windows")
    macos = write("macos.json", "macOS")
    stale_plan = write("old-plan.json", "Linux", test_plan_digest="0" * 64)
    with pytest.raises(EvidenceProductionError, match="stale"):
        aggregate_core_markers(
            "current",
            [stale_plan, windows, macos],
            tmp_path / "out.json",
            ci_run_id="run-1",
            ci_provider="https://github.com",
        )
    duplicate = write("linux-duplicate.json", "Linux")
    with pytest.raises(EvidenceProductionError, match="exactly three|duplicate"):
        aggregate_core_markers(
            "current",
            [linux, duplicate, windows, macos],
            tmp_path / "out.json",
            ci_run_id="run-1",
            ci_provider="https://github.com",
        )
    wrong_run = write("wrong-run.json", "Linux", ci_run_id="run-old")
    with pytest.raises(EvidenceProductionError, match="stale"):
        aggregate_core_markers(
            "current",
            [wrong_run, windows, macos],
            tmp_path / "out.json",
            ci_run_id="run-1",
            ci_provider="https://github.com",
        )
