from __future__ import annotations

import ast
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from deployment.platform_readiness import blocking_items, load_contract, production_ready
from deployment.platforms.windows import (
    SERVICE_IDENTITY,
    WindowsDeploymentNotQualified,
    qualify_acl,
    static_path_layout,
)


def test_canonical_platforms_and_reference() -> None:
    data = load_contract()
    assert data["platforms"] == ["WINDOWS", "LINUX", "MACOS"]
    assert data["reference_platform"] == "WINDOWS"
    assert set(data["allowed_statuses"]) == {
        "PASS", "FAIL", "NOT_IMPLEMENTED", "NOT_TESTED",
        "UNVERIFIED_ENVIRONMENT_LIMITATION", "NOT_APPLICABLE",
    }
    assert SERVICE_IDENTITY == r"NT SERVICE\CryptoHunterBackend"


def test_release_gates_are_platform_isolated_and_fail_closed() -> None:
    data = load_contract()
    gates = data["release_gates"]
    assert not any(item.startswith(("LINUX_", "MACOS_")) for item in gates["WINDOWS_PRODUCTION_READY"])
    assert not any(item.startswith("WINDOWS_") for item in gates["LINUX_PRODUCTION_READY"])
    assert not any(item.startswith(("WINDOWS_", "LINUX_")) for item in gates["MACOS_PRODUCTION_READY"])
    assert production_ready("WINDOWS", data) is False

    self_attested = copy.deepcopy(data)
    for item in self_attested["release_gates"]["WINDOWS_PRODUCTION_READY"]:
        self_attested["acceptance"][item]["status"] = "PASS"
    assert production_ready("WINDOWS", self_attested, [], "revision-1") is False


def test_required_core_failure_blocks_every_platform() -> None:
    data = load_contract()
    for item in data["acceptance"].values():
        item["status"] = "PASS"
    data["acceptance"]["CORE_REQUIRED_SUITES"]["status"] = "FAIL"
    assert all(not production_ready(platform, data, [], "revision-1") for platform in data["platforms"])


def test_windows_static_path_layout_does_not_claim_native_integration() -> None:
    paths = static_path_layout(
        r"C:\Program Files", r"C:\ProgramData", r"C:\Users\u\AppData\Local"
    )
    assert str(paths[0]) == r"C:\Program Files\CryptoHunter"
    assert str(paths[1]) == r"C:\ProgramData\CryptoHunter\State"
    data = load_contract()
    assert data["acceptance"]["WINDOWS_PATH_LAYOUT_STATIC_CONTRACT"]["status"] == "PASS"
    assert data["acceptance"]["WINDOWS_NATIVE_PATH_INTEGRATION"]["status"] == "UNVERIFIED_ENVIRONMENT_LIMITATION"
    with pytest.raises(WindowsDeploymentNotQualified, match="DACL"):
        qualify_acl()


def test_release_gate_cli_is_negative_now_and_positive_only_for_complete_evidence(
    tmp_path: Path,
) -> None:
    revision = "revision-1"
    command = [
        sys.executable, "-m", "deployment.platform_readiness", "--platform", "WINDOWS",
        "--current-revision", revision, "--expected-ci-provider", "test",
        "--expected-ci-run-id", "1",
    ]
    current = subprocess.run(command, capture_output=True, text=True, check=False)
    assert current.returncode != 0
    assert "WINDOWS_PRODUCTION_READY=NOT_READY" in current.stdout

    data = load_contract()
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(data), encoding="utf-8")
    windows_evidence, core_evidence = _complete_windows_evidence(data, revision)
    windows_path = tmp_path / "windows.json"
    core_path = tmp_path / "core.json"
    windows_path.write_text(json.dumps(windows_evidence), encoding="utf-8")
    core_path.write_text(json.dumps(core_evidence), encoding="utf-8")
    ready = subprocess.run(
        [*command, "--contract", str(contract_path), "--evidence", str(windows_path),
         "--evidence", str(core_path)], capture_output=True, text=True, check=False
    )
    assert ready.returncode == 0
    assert "WINDOWS_PRODUCTION_READY=PASS" in ready.stdout


def _document(platform_name: str, runner_os: str, revision: str, results: list[dict[str, str]]) -> dict[str, object]:
    return {
        "schema_version": 1, "platform": platform_name, "source_revision": revision,
        "ci_provider": "test", "ci_run_id": "1", "runner_os": runner_os,
        "runner_arch": "x64", "generated_at_utc": "2026-01-01T00:00:00Z",
        "results": results,
    }


def _result(item: str, evidence_class: str, status: str = "PASS") -> dict[str, str]:
    return {"item": item, "status": status, "evidence_class": evidence_class,
            "test_or_probe": "test probe", "details": "executed"}


def _complete_windows_evidence(data: dict[str, object], revision: str) -> tuple[dict[str, object], dict[str, object]]:
    acceptance = data["acceptance"]
    required = data["release_gates"]["WINDOWS_PRODUCTION_READY"]
    live = [_result(item, "LIVE_WINDOWS_INTEGRATION") for item in required
            if acceptance[item]["evidence_class"] == "LIVE_WINDOWS_INTEGRATION"]
    core = [_result("CORE_REQUIRED_SUITES", "CROSS_OS_CI_MATRIX")]
    return _document("WINDOWS", "Windows", revision, live), _document("CROSS_PLATFORM_CORE", "Windows", revision, core)


def test_current_windows_evidence_is_accepted_only_when_complete() -> None:
    data = load_contract()
    windows, core = _complete_windows_evidence(data, "current")
    assert production_ready("WINDOWS", data, [windows, core], "current") is True
    windows["results"].pop()
    assert production_ready("WINDOWS", data, [windows, core], "current") is False


def test_current_scm_slice_evidence_satisfies_only_its_exact_items() -> None:
    data = load_contract()
    scm_items = {
        "WINDOWS_SERVICE_INSTALLATION", "WINDOWS_SERVICE_START",
        "WINDOWS_GRACEFUL_STOP", "WINDOWS_MANUAL_RESTART",
    }
    windows = _document(
        "WINDOWS", "Windows", "current",
        [_result(item, "LIVE_WINDOWS_INTEGRATION") for item in sorted(scm_items)],
    )
    core = _document(
        "CROSS_PLATFORM_CORE", "Windows", "current",
        [_result("CORE_REQUIRED_SUITES", "CROSS_OS_CI_MATRIX")],
    )
    blockers = blocking_items("WINDOWS", data, [windows, core], "current")
    assert scm_items.isdisjoint(blockers)
    assert "WINDOWS_CLEAN_INSTALL" in blockers
    assert production_ready("WINDOWS", data, [windows, core], "current") is False


def test_stale_wrong_os_duplicate_unknown_and_unsupported_evidence_fail_closed() -> None:
    data = load_contract()
    windows, core = _complete_windows_evidence(data, "current")
    stale = copy.deepcopy(windows)
    stale["source_revision"] = "previous"
    assert production_ready("WINDOWS", data, [stale, core], "current") is False
    wrong_os = copy.deepcopy(windows)
    wrong_os["runner_os"] = "Linux"
    assert production_ready("WINDOWS", data, [wrong_os, core], "current") is False
    duplicate = copy.deepcopy(windows)
    duplicate["results"].append(copy.deepcopy(duplicate["results"][0]))
    duplicate["results"][-1]["status"] = "FAIL"
    assert production_ready("WINDOWS", data, [duplicate, core], "current") is False
    unknown = copy.deepcopy(windows)
    unknown["results"][0]["item"] = "UNKNOWN_ITEM"
    assert production_ready("WINDOWS", data, [unknown, core], "current") is False
    unsupported = copy.deepcopy(windows)
    unsupported["results"][0]["evidence_class"] = "SELF_ATTESTED"
    assert production_ready("WINDOWS", data, [unsupported, core], "current") is False
    wrong_run = copy.deepcopy(windows)
    wrong_run["ci_run_id"] = "previous-run"
    assert production_ready("WINDOWS", data, [wrong_run, core], "current") is False
    wrong_provider = copy.deepcopy(windows)
    wrong_provider["ci_provider"] = "anything"
    assert production_ready("WINDOWS", data, [wrong_provider, core], "current") is False


@pytest.mark.parametrize("malformed", [{}, {"schema_version": 1}, {"results": "not-a-list"}])
def test_malformed_evidence_fails_closed(malformed: dict[str, object]) -> None:
    assert production_ready("WINDOWS", load_contract(), [malformed], "current") is False


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def test_architecture_import_separation() -> None:
    root = Path(__file__).resolve().parents[2]
    forbidden = ("deployment.platforms", "deployment.systemd")
    for folder in (root / "core", root / "bot_core"):
        for source in folder.rglob("*.py"):
            assert not any(name.startswith(forbidden) for name in _imports(source)), source
    assert not any("systemd" in name for name in _imports(root / "deployment/platforms/windows.py"))
    assert not any("systemd" in name for name in _imports(root / "deployment/platforms/macos.py"))


def test_legacy_security_flags_remain_false() -> None:
    flags = load_contract()["legacy_flags"]
    for name in ("FreshnessAuthority_implemented", "ROOT_PROOF_ISSUER_IMPLEMENTED", "production_substrate_implemented", "PRODUCTION_LOCAL_RUNTIME_AVAILABLE"):
        assert flags[name] is False


def test_every_pass_has_executable_evidence_class() -> None:
    data = load_contract()
    passed = {name: item for name, item in data["acceptance"].items() if item["status"] == "PASS"}
    assert set(passed) == {
        "WINDOWS_PATH_LAYOUT_STATIC_CONTRACT",
        "LINUX_STATIC_QUALIFICATION",
        "LINUX_INSTALLER_STATIC_CONTRACT",
    }
    assert all(item["evidence_class"] == "STATIC_EXECUTABLE_CONTRACT" for item in passed.values())


def test_workflow_release_gate_is_real_and_platform_independent() -> None:
    workflow = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / ".github/workflows/platform-deployment.yml").read_text(
            encoding="utf-8"
        )
    )
    jobs = workflow["jobs"]
    core_manifest = json.loads(
        (Path(__file__).resolve().parents[2] / "deployment/core_required_suites_v1.json").read_text(
            encoding="utf-8"
        )
    )
    for job_name in ("core-linux", "core-windows", "core-macos"):
        commands = [step.get("run", "") for step in jobs[job_name]["steps"]]
        assert core_manifest["optional_dependency_setup"] in commands
        assert any("python -m deployment.core_test_plan" in command for command in commands)
        assert not any("platform_evidence core-marker" in command for command in commands)
    release = jobs["windows-release-gate"]
    assert set(release["needs"]) == {
        "core-linux", "core-windows", "core-macos",
        "windows-deployment-contract", "windows-deployment-integration",
    }
    assert "linux-deployment-integration" not in release["needs"]
    assert "macos-deployment-integration" not in release["needs"]
    release_commands = [step.get("run", "") for step in release["steps"]]
    assert any("python -m deployment.platform_readiness --platform WINDOWS" in command
               and "--evidence evidence/core-matrix.json" in command
               and "--evidence evidence/windows-scm-evidence.json" in command
               for command in release_commands)
    integration_commands = [
        step.get("run", "") for step in jobs["windows-deployment-integration"]["steps"]
    ]
    assert any("python -m deployment.windows_acceptance --mode github" in command
               for command in integration_commands)
    assert not any("windows_scm_probe.ps1" in command for command in integration_commands)
    assert not any(command.lstrip().startswith("echo ") for command in integration_commands)
    assert any(step.get("uses", "").startswith("actions/upload-artifact")
               for step in jobs["windows-deployment-integration"]["steps"])
