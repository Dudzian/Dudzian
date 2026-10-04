from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from deployment import platform_evidence
from deployment.platform_evidence import (
    EvidenceProductionError,
    aggregate_core_markers,
    produce_windows_clean_install_evidence,
    produce_windows_stage10_evidence,
)
from deployment.platform_readiness import blocking_items, load_contract, production_ready
from deployment.windows_stage9_evidence_contract import PRODUCTION_TRUST_CEREMONY_ID

CEREMONY_ID = PRODUCTION_TRUST_CEREMONY_ID
PUBLIC_PACKAGE_FILENAMES = {"package_manifest.json"}
CANONICAL_TEST_PACKAGE_MANIFEST = b"{}\n"


def _write_test_production_trust_package(parent: Path) -> Path:
    trust_package = parent / CEREMONY_ID
    trust_package.mkdir()
    for name in PUBLIC_PACKAGE_FILENAMES:
        (trust_package / name).write_bytes(CANONICAL_TEST_PACKAGE_MANIFEST)
    return trust_package


def _stage10_receipt() -> dict[str, object]:
    contract = json.loads(platform_evidence.WINDOWS_STAGE10_CONTRACT.read_text())
    proof_names = contract["results"]
    return {
        "schema_version": 1,
        "source_revision": "revision",
        "ci_provider": "https://github.com",
        "ci_run_id": "42",
        "runner_os": "Windows",
        "runner_arch": "X64",
        "runner_name": "stage10-host",
        "machine_guid": "machine-a",
        "probe_id": "cryptohunter.windows.stage10.lifecycle.v1",
        "runtime_provenance": {
            "source_revision": "revision",
            "ci_run_id": "42",
            "product_version": "1.2.3",
            "qualified_manifest_sha256": "a" * 64,
            "qualified_msi_sha256": "b" * 64,
            "installed_backend_sha256": "c" * 64,
        },
        "results": {
            item: {
                "status": "PASS",
                "proofs": {p: "PASS" for p in proofs},
                "details": f"native proof for {item}",
            }
            for item, proofs in proof_names.items()
        },
    }


def test_stage10_current_native_receipt_is_consumable(tmp_path, monkeypatch) -> None:
    for key, value in {
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_RUN_ID": "42",
        "RUNNER_OS": "Windows",
        "RUNNER_ARCH": "X64",
        "RUNNER_NAME": "stage10-host",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(platform_evidence, "_current_windows_machine_guid", lambda: "machine-a")
    receipt, output = tmp_path / "receipt.json", tmp_path / "evidence.json"
    receipt.write_text(json.dumps(_stage10_receipt()), encoding="utf-8")
    produce_windows_stage10_evidence("revision", receipt, output)
    results = json.loads(output.read_text())["results"]
    assert {r["item"] for r in results} == set(_stage10_receipt()["results"])
    assert all(
        r["status"] == "PASS" and r["evidence_class"] == "LIVE_WINDOWS_INTEGRATION" for r in results
    )
    contract = load_contract()
    contract["release_gates"]["WINDOWS_PRODUCTION_READY"] = list(_stage10_receipt()["results"])
    assert production_ready(
        "WINDOWS",
        contract,
        [json.loads(output.read_text())],
        "revision",
        "https://github.com",
        "42",
    )


def test_canonical_contract_accepts_lifecycle_but_keeps_update_gates_blocking(
    tmp_path,
    monkeypatch,
) -> None:
    for key, value in {
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_RUN_ID": "42",
        "RUNNER_OS": "Windows",
        "RUNNER_ARCH": "X64",
        "RUNNER_NAME": "stage10-host",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(platform_evidence, "_current_windows_machine_guid", lambda: "machine-a")
    receipt, output = tmp_path / "receipt.json", tmp_path / "evidence.json"
    receipt.write_text(json.dumps(_stage10_receipt()))
    produce_windows_stage10_evidence("revision", receipt, output)
    evidence = json.loads(output.read_text())
    data = load_contract()
    blockers = blocking_items("WINDOWS", data, [evidence], "revision", "https://github.com", "42")
    assert "WINDOWS_REBOOT_RECOVERY" not in blockers
    assert "WINDOWS_LONG_RUNNING_LIFECYCLE" not in blockers
    assert "WINDOWS_UPDATE_RESTART" in blockers
    assert "WINDOWS_FAILED_UPDATE_SAFE_BEHAVIOR" in blockers
    assert not production_ready("WINDOWS", data, [evidence], "revision", "https://github.com", "42")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(source_revision="stale"),
        lambda r: r.update(ci_run_id="old-run"),
        lambda r: r.update(ci_provider="untrusted"),
        lambda r: r.update(runner_name="other"),
        lambda r: r.update(runner_arch="ARM64"),
        lambda r: r.update(machine_guid="machine-b"),
        lambda r: r.pop("runtime_provenance"),
        lambda r: r["runtime_provenance"].update(qualified_manifest_sha256="d" * 63),
        lambda r: r["runtime_provenance"].update(qualified_msi_sha256="d" * 63),
        lambda r: r["results"].pop("WINDOWS_REBOOT_RECOVERY"),
        lambda r: r["results"].update(EXTRA_RESULT={}),
        lambda r: r["results"]["WINDOWS_REBOOT_RECOVERY"].update(status="FAIL"),
        (
            lambda r: r["results"]["WINDOWS_LONG_RUNNING_LIFECYCLE"]["proofs"].pop(
                "readiness_contract_before_after"
            )
        ),
    ],
)
def test_stage10_rejects_stale_missing_or_failed_receipt(tmp_path, monkeypatch, mutation) -> None:
    for key, value in {
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_RUN_ID": "42",
        "RUNNER_OS": "Windows",
        "RUNNER_ARCH": "X64",
        "RUNNER_NAME": "stage10-host",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(platform_evidence, "_current_windows_machine_guid", lambda: "machine-a")
    value = _stage10_receipt()
    mutation(value)
    _write_test_production_trust_package(tmp_path)

    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(value))
    with pytest.raises(EvidenceProductionError):
        produce_windows_stage10_evidence("revision", receipt, tmp_path / "out.json")


@pytest.mark.parametrize("observed", [None, "", 7])
def test_stage10_rejects_missing_empty_or_wrong_type_machine_guid(
    tmp_path,
    monkeypatch,
    observed,
) -> None:
    for key, value in {
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_RUN_ID": "42",
        "RUNNER_OS": "Windows",
        "RUNNER_ARCH": "X64",
        "RUNNER_NAME": "stage10-host",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(platform_evidence, "_current_windows_machine_guid", lambda: observed)
    _write_test_production_trust_package(tmp_path)

    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(_stage10_receipt()))
    with pytest.raises(EvidenceProductionError):
        produce_windows_stage10_evidence("revision", receipt, tmp_path / "out.json")


def test_machine_guid_reader_fails_closed_when_registry_key_is_missing(monkeypatch) -> None:
    class MissingWinreg:
        HKEY_LOCAL_MACHINE = object()

        @staticmethod
        def OpenKey(*_args):
            raise FileNotFoundError("missing MachineGuid key")

    monkeypatch.setitem(__import__("sys").modules, "winreg", MissingWinreg)
    with pytest.raises(EvidenceProductionError, match="MachineGuid is unavailable"):
        platform_evidence._current_windows_machine_guid()


@pytest.mark.parametrize("probe_id", [None, "", "cryptohunter.windows.stage10.other.v1"])
def test_stage10_rejects_invalid_shared_contract_probe_id(
    tmp_path,
    monkeypatch,
    probe_id,
) -> None:
    for key, value in {
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_RUN_ID": "42",
        "RUNNER_OS": "Windows",
        "RUNNER_ARCH": "X64",
        "RUNNER_NAME": "stage10-host",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(platform_evidence, "_current_windows_machine_guid", lambda: "machine-a")
    contract = json.loads(platform_evidence.WINDOWS_STAGE10_CONTRACT.read_text())
    contract["probe_id"] = probe_id
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    _write_test_production_trust_package(tmp_path)

    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(_stage10_receipt()))
    with pytest.raises(EvidenceProductionError, match="lifecycle contract"):
        produce_windows_stage10_evidence(
            "revision", receipt, tmp_path / "out.json", contract_path=contract_path
        )


from deployment.core_test_plan import load_manifest, marker_document
from deployment.windows_stage9_evidence_contract import (
    CLEAN_INSTALL_PRECEREMONY_PROOFS,
    POST_ENROLLMENT_QUALIFICATION_STATE,
)


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


def _current_clean_install_receipt(msi_hash: str, manifest_hash: str) -> dict[str, object]:
    """Build the schema emitted by the Stage-9 producer without simulating Windows."""
    return {
        "schema_version": 1,
        "source_revision": "revision",
        "ci_provider": "https://github.com",
        "ci_run_id": "42",
        "runner_os": "Windows",
        "probe_id": "cryptohunter.windows.clean-install.v1",
        "msi_sha256": msi_hash,
        "manifest_sha256": manifest_hash,
        "product_version": "1.2.3",
        "runner_arch": "X64",
        "production_trust_ceremony_id": CEREMONY_ID,
        "production_trust_package_manifest_sha256": hashlib.sha256(
            CANONICAL_TEST_PACKAGE_MANIFEST
        ).hexdigest(),
        "production_trust_artifact_run_id": "37216551602",
        "install_exit_code": 0,
        "uninstall_exit_code": 0,
        "proofs": {name: "PASS" for name in CLEAN_INSTALL_PRECEREMONY_PROOFS},
        "post_enrollment_live_qualification": POST_ENROLLMENT_QUALIFICATION_STATE,
    }


@pytest.fixture
def clean_install_contract_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path]:
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("RUNNER_OS", "Windows")
    monkeypatch.setenv("RUNNER_ARCH", "X64")
    msi = tmp_path / "product.msi"
    manifest = tmp_path / "installer-manifest.json"
    msi.write_bytes(b"msi")
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

    _write_test_production_trust_package(tmp_path)

    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            _current_clean_install_receipt(
                msi_hash, hashlib.sha256(manifest.read_bytes()).hexdigest()
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "evidence.json"
    return receipt, output, msi, manifest


def test_production_trust_fixture_uses_canonical_manifest_bytes(
    clean_install_contract_files: tuple[Path, Path, Path, Path],
) -> None:
    receipt, _, _, _ = clean_install_contract_files
    package_manifest = receipt.parent / CEREMONY_ID / "package_manifest.json"
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))

    assert package_manifest.read_bytes() == b"{}\n"
    assert (
        hashlib.sha256(package_manifest.read_bytes()).hexdigest()
        == (receipt_value["production_trust_package_manifest_sha256"])
    )


def test_current_producer_receipt_is_accepted_by_evidence_consumer(
    clean_install_contract_files: tuple[Path, Path, Path, Path],
) -> None:
    receipt, output, msi, manifest = clean_install_contract_files
    produce_windows_clean_install_evidence(
        "revision",
        receipt,
        output,
        msi=msi,
        manifest=manifest,
        production_trust_package=receipt.parent / CEREMONY_ID,
    )
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["results"][0]["item"] == "WINDOWS_CLEAN_INSTALL"
    assert evidence["results"][0]["status"] == "PASS"
    assert evidence["results"][0]["evidence_class"] == "LIVE_WINDOWS_INTEGRATION"
    contract = load_contract()
    contract["release_gates"]["WINDOWS_PRODUCTION_READY"] = ["WINDOWS_CLEAN_INSTALL"]
    assert production_ready(
        "WINDOWS",
        contract,
        [evidence],
        "revision",
        "https://github.com",
        "42",
    )


def _write_manifest_and_rebind_receipt(
    manifest: Path, receipt: Path, manifest_value: dict[str, object]
) -> None:
    manifest.write_text(json.dumps(manifest_value), encoding="utf-8")
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_value["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(receipt_value), encoding="utf-8")


def test_clean_install_evidence_accepts_wix_build_metadata(
    clean_install_contract_files: tuple[Path, Path, Path, Path],
) -> None:
    receipt, output, msi, manifest = clean_install_contract_files
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_value["wix_version"] = "7.0.0+b8977d6"
    _write_manifest_and_rebind_receipt(manifest, receipt, manifest_value)

    produce_windows_clean_install_evidence(
        "revision",
        receipt,
        output,
        msi=msi,
        manifest=manifest,
        production_trust_package=receipt.parent / CEREMONY_ID,
    )
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["results"][0]["status"] == "PASS"


@pytest.mark.parametrize(
    "wix_version",
    ["7.0.1", "7.0.0-preview.1", "7.0.01", "7.0.0 unexpected", "7.0.0+", None],
)
def test_clean_install_evidence_rejects_unreviewed_wix_version(
    clean_install_contract_files: tuple[Path, Path, Path, Path],
    wix_version: object,
) -> None:
    receipt, output, msi, manifest = clean_install_contract_files
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_value["wix_version"] = wix_version
    _write_manifest_and_rebind_receipt(manifest, receipt, manifest_value)

    with pytest.raises(EvidenceProductionError, match="clean-install receipt"):
        produce_windows_clean_install_evidence(
            "revision",
            receipt,
            output,
            msi=msi,
            manifest=manifest,
            production_trust_package=receipt.parent / CEREMONY_ID,
        )


def test_clean_install_evidence_rejects_missing_wix_version(
    clean_install_contract_files: tuple[Path, Path, Path, Path],
) -> None:
    receipt, output, msi, manifest = clean_install_contract_files
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_value.pop("wix_version")
    _write_manifest_and_rebind_receipt(manifest, receipt, manifest_value)

    with pytest.raises(EvidenceProductionError, match="clean-install receipt"):
        produce_windows_clean_install_evidence(
            "revision",
            receipt,
            output,
            msi=msi,
            manifest=manifest,
            production_trust_package=receipt.parent / CEREMONY_ID,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("post_enrollment_live_qualification"),
        lambda value: value.update(post_enrollment_live_qualification="OPTIONAL"),
        lambda value: value["proofs"].pop("authority_absent"),
        lambda value: value["proofs"].pop("frozen_production_trust"),
        lambda value: value["proofs"].update(frozen_production_trust="FAIL"),
        lambda value: value.update(production_trust_ceremony_id="0" * 64),
        lambda value: value.update(production_trust_package_manifest_sha256="0" * 64),
        lambda value: value["proofs"].pop("production_enrollment_fail_closed"),
        lambda value: (
            value["proofs"].pop("authority_absent"),
            value["proofs"].update(mtls_matrix="PASS"),
        ),
        lambda value: (
            value["proofs"].pop("production_enrollment_fail_closed"),
            value["proofs"].update(backend="PASS", logging="PASS"),
        ),
        lambda value: value.update(unknown="PASS"),
        lambda value: value["proofs"].update(files="FAIL"),
    ],
    ids=[
        "missing-qualification",
        "wrong-qualification",
        "missing-authority-absent",
        "missing-frozen-trust",
        "failed-frozen-trust",
        "wrong-trust-ceremony",
        "wrong-trust-manifest-sha",
        "missing-enrollment-fail-closed",
        "mtls-substitution",
        "backend-logging-substitution",
        "unknown-receipt-key",
        "failed-proof",
    ],
)
def test_clean_install_evidence_rejects_contract_drift(
    clean_install_contract_files: tuple[Path, Path, Path, Path],
    mutation: Callable[[dict[str, Any]], object],
) -> None:
    receipt, output, msi, manifest = clean_install_contract_files
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    mutation(receipt_value)
    receipt.write_text(json.dumps(receipt_value), encoding="utf-8")
    with pytest.raises(EvidenceProductionError, match="clean-install receipt"):
        produce_windows_clean_install_evidence(
            "revision",
            receipt,
            output,
            msi=msi,
            manifest=manifest,
            production_trust_package=receipt.parent / CEREMONY_ID,
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
