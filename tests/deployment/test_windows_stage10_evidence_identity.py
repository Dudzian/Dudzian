"""Synthetic receipt checks; never execute the native Stage-10 qualification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from deployment import platform_evidence
from deployment.platform_evidence import EvidenceProductionError, produce_windows_stage10_evidence
from deployment.platform_readiness import blocking_items, load_contract


LIFECYCLE_PROOFS = (
    ("WINDOWS_REBOOT_RECOVERY", "boot_changed"),
    ("WINDOWS_REBOOT_RECOVERY", "current_revision_runtime_bound"),
    ("WINDOWS_REBOOT_RECOVERY", "service_autostarted"),
    ("WINDOWS_REBOOT_RECOVERY", "readiness_contract_after_reboot"),
    ("WINDOWS_LONG_RUNNING_LIFECYCLE", "duration_at_least_24h"),
    ("WINDOWS_LONG_RUNNING_LIFECYCLE", "current_revision_runtime_bound"),
    ("WINDOWS_LONG_RUNNING_LIFECYCLE", "same_service_instance"),
    ("WINDOWS_LONG_RUNNING_LIFECYCLE", "readiness_contract_before_after"),
)


@pytest.fixture
def native_receipt_identity(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    for name, value in {
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_RUN_ID": "42",
        "RUNNER_OS": "Windows",
        "RUNNER_ARCH": "X64",
        "RUNNER_NAME": "stage10-test-host",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(platform_evidence, "_current_windows_machine_guid", lambda: "test-machine")
    contract = json.loads(platform_evidence.WINDOWS_STAGE10_CONTRACT.read_text())
    return {
        "schema_version": 1,
        "source_revision": "test-revision",
        "ci_provider": "https://github.com",
        "ci_run_id": "42",
        "runner_os": "Windows",
        "runner_arch": "X64",
        "runner_name": "stage10-test-host",
        "machine_guid": "test-machine",
        "probe_id": contract["probe_id"],
        "runtime_provenance": {
            "source_revision": "test-revision",
            "ci_run_id": "42",
            "product_version": "1.2.3",
            "qualified_manifest_sha256": "a" * 64,
            "qualified_msi_sha256": "b" * 64,
            "installed_backend_sha256": "c" * 64,
        },
        "results": {
            item: {
                "status": "PASS",
                "proofs": {proof: "PASS" for proof in proofs},
                "details": f"synthetic test fixture: {item}",
            }
            for item, proofs in contract["results"].items()
        },
    }


def _receipt_file(tmp_path: Path, receipt: dict[str, Any]) -> Path:
    path = tmp_path / "synthetic-stage10-receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


@pytest.mark.parametrize("version", [True, 1.0])
def test_stage10_rejects_noninteger_receipt_version(
    tmp_path: Path, native_receipt_identity: dict[str, Any], version: object
) -> None:
    native_receipt_identity["schema_version"] = version
    output = tmp_path / "evidence.json"
    with pytest.raises(EvidenceProductionError, match="identity"):
        produce_windows_stage10_evidence(
            "test-revision", _receipt_file(tmp_path, native_receipt_identity), output
        )
    assert not output.exists()


@pytest.mark.parametrize("version", [True, 1.0])
def test_stage10_rejects_noninteger_lifecycle_contract_version(
    tmp_path: Path, native_receipt_identity: dict[str, Any], version: object
) -> None:
    contract = json.loads(platform_evidence.WINDOWS_STAGE10_CONTRACT.read_text())
    contract["schema_version"] = version
    contract_path = tmp_path / "lifecycle-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    output = tmp_path / "evidence.json"
    with pytest.raises(EvidenceProductionError, match="lifecycle contract"):
        produce_windows_stage10_evidence(
            "test-revision",
            _receipt_file(tmp_path, native_receipt_identity),
            output,
            contract_path=contract_path,
        )
    assert not output.exists()


@pytest.mark.parametrize("field", ["runner_name", "runner_arch"])
@pytest.mark.parametrize("identity", [None, "", " \t"])
def test_stage10_rejects_matching_but_missing_current_runner_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_receipt_identity: dict[str, Any],
    field: str,
    identity: str | None,
) -> None:
    environment_name = field.upper()
    if identity is None:
        monkeypatch.delenv(environment_name)
    else:
        monkeypatch.setenv(environment_name, identity)
    native_receipt_identity[field] = identity
    output = tmp_path / "evidence.json"
    with pytest.raises(EvidenceProductionError, match="current runner identity"):
        produce_windows_stage10_evidence(
            "test-revision", _receipt_file(tmp_path, native_receipt_identity), output
        )
    assert not output.exists()


def test_missing_stage10_receipt_refuses_without_output(
    tmp_path: Path, native_receipt_identity: dict[str, Any]
) -> None:
    output = tmp_path / "evidence.json"
    assert (
        platform_evidence.main(
            [
                "windows-stage10",
                "--source-revision",
                "test-revision",
                "--receipt",
                str(tmp_path / "absent-stage10.json"),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert not output.exists()


def test_lifecycle_receipt_cannot_supply_lifecycle_or_update_qualification(
    tmp_path: Path, native_receipt_identity: dict[str, Any]
) -> None:
    output = tmp_path / "evidence.json"
    with pytest.raises(
        EvidenceProductionError, match="WINDOWS_STAGE10_NATIVE_PROOF_AUTHORITY_UNAVAILABLE"
    ):
        produce_windows_stage10_evidence(
            "test-revision", _receipt_file(tmp_path, native_receipt_identity), output
        )
    assert not output.exists()
    blockers = blocking_items(
        "WINDOWS", load_contract(), [], "test-revision", "https://github.com", "42"
    )
    assert "WINDOWS_REBOOT_RECOVERY" in blockers
    assert "WINDOWS_LONG_RUNNING_LIFECYCLE" in blockers
    assert "WINDOWS_UPDATE_RESTART" in blockers
    assert "WINDOWS_FAILED_UPDATE_SAFE_BEHAVIOR" in blockers
    assert "CORE_REQUIRED_SUITES" in blockers


def test_lifecycle_proof_regressions_cover_the_complete_contract() -> None:
    contract = json.loads(platform_evidence.WINDOWS_STAGE10_CONTRACT.read_text())
    assert set(LIFECYCLE_PROOFS) == {
        (item, proof) for item, proofs in contract["results"].items() for proof in proofs
    }


@pytest.mark.parametrize(("item", "proof"), LIFECYCLE_PROOFS)
@pytest.mark.parametrize("status", ["MISSING", "UNKNOWN", None, True, "PASS"])
def test_each_lifecycle_declaration_refuses_without_native_authority(
    tmp_path: Path,
    native_receipt_identity: dict[str, Any],
    item: str,
    proof: str,
    status: object,
) -> None:
    proofs = native_receipt_identity["results"][item]["proofs"]
    if status == "MISSING":
        proofs.pop(proof)
    else:
        proofs[proof] = status
    output = tmp_path / "evidence.json"
    expected = (
        "WINDOWS_STAGE10_NATIVE_PROOF_AUTHORITY_UNAVAILABLE"
        if status == "PASS"
        else f"Stage-10 receipt failed: {item}"
    )
    with pytest.raises(EvidenceProductionError, match=expected):
        produce_windows_stage10_evidence(
            "test-revision", _receipt_file(tmp_path, native_receipt_identity), output
        )
    assert not output.exists()


def test_cli_cannot_publish_synthetic_lifecycle_pass(
    tmp_path: Path,
    native_receipt_identity: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "evidence.json"
    receipt = _receipt_file(tmp_path, native_receipt_identity)
    assert (
        platform_evidence.main(
            [
                "windows-stage10",
                "--source-revision",
                "test-revision",
                "--receipt",
                str(receipt),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert "WINDOWS_STAGE10_NATIVE_PROOF_AUTHORITY_UNAVAILABLE" in capsys.readouterr().err
    assert not output.exists()


def test_refused_receipt_invalidates_stale_output(
    tmp_path: Path, native_receipt_identity: dict[str, Any]
) -> None:
    output = tmp_path / "evidence.json"
    output.write_text('{"results": [{"status": "PASS"}]}', encoding="utf-8")
    with pytest.raises(
        EvidenceProductionError, match="WINDOWS_STAGE10_NATIVE_PROOF_AUTHORITY_UNAVAILABLE"
    ):
        produce_windows_stage10_evidence(
            "test-revision", _receipt_file(tmp_path, native_receipt_identity), output
        )
    assert not output.exists()


@pytest.mark.parametrize("alias", ["receipt", "contract"])
def test_output_alias_refuses_without_deleting_input(
    tmp_path: Path, native_receipt_identity: dict[str, Any], alias: str
) -> None:
    receipt = _receipt_file(tmp_path, native_receipt_identity)
    contract = tmp_path / "lifecycle-contract.json"
    contract.write_bytes(platform_evidence.WINDOWS_STAGE10_CONTRACT.read_bytes())
    output = receipt if alias == "receipt" else contract
    original = output.read_bytes()
    with pytest.raises(EvidenceProductionError):
        produce_windows_stage10_evidence("test-revision", receipt, output, contract_path=contract)
    assert output.read_bytes() == original


@pytest.mark.parametrize(("item", "proof"), LIFECYCLE_PROOFS)
def test_weakened_contract_cannot_omit_a_required_observation(
    tmp_path: Path, native_receipt_identity: dict[str, Any], item: str, proof: str
) -> None:
    contract = json.loads(platform_evidence.WINDOWS_STAGE10_CONTRACT.read_text())
    contract["results"][item].remove(proof)
    native_receipt_identity["results"][item]["proofs"].pop(proof)
    path = tmp_path / "weakened-contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    output = tmp_path / "evidence.json"
    with pytest.raises(EvidenceProductionError, match="lifecycle contract"):
        produce_windows_stage10_evidence(
            "test-revision",
            _receipt_file(tmp_path, native_receipt_identity),
            output,
            contract_path=path,
        )
    assert not output.exists()
