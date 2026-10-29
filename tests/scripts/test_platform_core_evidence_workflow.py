"""The Linux gate consumes the complete current-run Core matrix without waivers."""

from pathlib import Path
import os
import shlex
import subprocess
import sys

import pytest
import yaml


def test_linux_gate_waits_for_and_consumes_all_current_run_core_artifacts() -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load(
        (root / ".github/workflows/platform-deployment.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["linux-deployment-integration"]
    assert set(job["needs"]) == {"core-linux", "core-windows", "core-macos"}
    assert not job.get("continue-on-error", False)
    downloads = {
        step["with"]["name"]: step
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/download-artifact@")
    }
    assert set(downloads) == {
        "core-evidence-Linux",
        "core-evidence-Windows",
        "core-evidence-macOS",
    }
    for step in downloads.values():
        assert not step.get("continue-on-error", False)
        # With no alternate run/repository/token, download-artifact uses this workflow run.
        assert not {"run-id", "repository", "github-token"} & set(step["with"])
    paths = {step["with"]["path"] for step in downloads.values()}
    assert len(paths) == 3

    aggregate = next(
        step for step in job["steps"] if "platform_evidence aggregate-core" in step.get("run", "")
    )
    arguments = shlex.split(aggregate["run"])
    markers = [arguments[index + 1] for index, value in enumerate(arguments) if value == "--marker"]
    assert set(markers) == {
        (
            Path(downloads[f"core-evidence-{os_name}"]["with"]["path"]) / f"core-{os_name}.json"
        ).as_posix()
        for os_name in ("Linux", "Windows", "macOS")
    }
    assert len(markers) == 3
    assert arguments[arguments.index("--source-revision") + 1] == "${{ github.sha }}"
    matrix = arguments[arguments.index("--output") + 1]
    assert not aggregate.get("continue-on-error", False)

    gate = next(step for step in job["steps"] if "platform_readiness" in step.get("run", ""))
    gate_arguments = shlex.split(gate["run"])
    assert gate_arguments[gate_arguments.index("--platform") + 1] == "LINUX"
    assert gate_arguments[gate_arguments.index("--evidence") + 1] == matrix
    assert gate_arguments[gate_arguments.index("--current-revision") + 1] == "${{ github.sha }}"
    assert (
        gate_arguments[gate_arguments.index("--expected-ci-run-id") + 1] == "${{ github.run_id }}"
    )
    assert (
        gate_arguments[gate_arguments.index("--expected-ci-provider") + 1]
        == "${{ github.server_url }}"
    )
    assert not gate.get("continue-on-error", False)
    assert job["steps"].index(aggregate) < job["steps"].index(gate)
    assert all(
        job["steps"].index(step) < job["steps"].index(aggregate) for step in downloads.values()
    )
    status_guard = next(step for step in job["steps"] if step.get("shell") == "python")
    assert status_guard["env"] == {
        "CORE_LINUX_RESULT": "${{ needs.core-linux.result }}",
        "CORE_WINDOWS_RESULT": "${{ needs.core-windows.result }}",
        "CORE_MACOS_RESULT": "${{ needs.core-macos.result }}",
    }
    assert not status_guard.get("continue-on-error", False)
    assert all(job["steps"].index(status_guard) < job["steps"].index(s) for s in downloads.values())


@pytest.mark.parametrize("current_result", ["success", "failure", "cancelled", "skipped", ""])
@pytest.mark.parametrize("system", ["LINUX", "WINDOWS", "MACOS"])
@pytest.mark.parametrize("job_name", ["linux-deployment-integration", "windows-release-gate"])
def test_current_core_status_guard_refuses_reusing_artifacts_after_failed_attempt(
    job_name: str, system: str, current_result: str
) -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/platform-deployment.yml").read_text())
    job = workflow["jobs"][job_name]
    guard = next(step for step in job["steps"] if step.get("shell") == "python")
    environment = dict(os.environ)
    environment.update({f"CORE_{name}_RESULT": "success" for name in ("LINUX", "WINDOWS", "MACOS")})
    environment[f"CORE_{system}_RESULT"] = current_result
    result = subprocess.run(
        [sys.executable, "-c", guard["run"]], env=environment, capture_output=True
    )
    assert (result.returncode == 0) is (current_result == "success")


def test_stage10_rechecks_final_service_identity_before_emitting_proofs() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "deployment/windows_stage10_live.ps1").read_text(encoding="utf-8")
    final_sample = source.index("$final = Service-Snapshot")
    final_guard = source.index(
        "Assert-Stage10SoakIdentity -Expected $soakIdentity -Observed $final"
    )
    proof_emission = source.index("$proofValues = @{}")
    assert final_sample < final_guard < proof_emission
    assert "windows_stage10_soak_identity.ps1" in source
    assert "Assert-Stage10SoakIdentity -Expected $soakIdentity -Observed $sample" in source

    workflow = yaml.safe_load(
        (root / ".github/workflows/platform-deployment.yml").read_text(encoding="utf-8")
    )
    step = next(
        step
        for step in workflow["jobs"]["windows-deployment-contract"]["steps"]
        if "windows_stage10_soak_identity_regression.ps1" in step.get("run", "")
    )
    assert step["shell"] == "powershell"
    assert not step.get("continue-on-error", False)
