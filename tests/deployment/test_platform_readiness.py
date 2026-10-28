from __future__ import annotations

import ast
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest
import yaml

from deployment.platform_readiness import blocking_items, load_contract, production_ready
from deployment.platforms.windows import (
    SERVICE_IDENTITY,
    WindowsDeploymentNotQualified,
    qualify_acl,
    static_path_layout,
)
from deployment.windows_stage9_evidence_contract import PRODUCTION_TRUST_CEREMONY_ID


def test_canonical_platforms_and_reference() -> None:
    data = load_contract()
    assert data["platforms"] == ["WINDOWS", "LINUX", "MACOS"]
    assert data["reference_platform"] == "WINDOWS"
    assert set(data["allowed_statuses"]) == {
        "PASS",
        "FAIL",
        "NOT_IMPLEMENTED",
        "NOT_TESTED",
        "UNVERIFIED_ENVIRONMENT_LIMITATION",
        "NOT_APPLICABLE",
    }
    assert SERVICE_IDENTITY == r"NT SERVICE\CryptoHunterBackend"


def test_release_gates_are_platform_isolated_and_fail_closed() -> None:
    data = load_contract()
    gates = data["release_gates"]
    assert not any(
        item.startswith(("LINUX_", "MACOS_")) for item in gates["WINDOWS_PRODUCTION_READY"]
    )
    assert not any(item.startswith("WINDOWS_") for item in gates["LINUX_PRODUCTION_READY"])
    assert not any(
        item.startswith(("WINDOWS_", "LINUX_")) for item in gates["MACOS_PRODUCTION_READY"]
    )
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
    assert all(
        not production_ready(platform, data, [], "revision-1") for platform in data["platforms"]
    )


def test_windows_static_path_layout_does_not_claim_native_integration() -> None:
    paths = static_path_layout(r"C:\Program Files", r"C:\ProgramData", r"C:\Users\u\AppData\Local")
    assert str(paths[0]) == r"C:\Program Files\CryptoHunter"
    assert str(paths[1]) == r"C:\ProgramData\CryptoHunter\State"
    data = load_contract()
    assert data["acceptance"]["WINDOWS_PATH_LAYOUT_STATIC_CONTRACT"]["status"] == "PASS"
    assert (
        data["acceptance"]["WINDOWS_NATIVE_PATH_INTEGRATION"]["status"]
        == "UNVERIFIED_ENVIRONMENT_LIMITATION"
    )
    with pytest.raises(WindowsDeploymentNotQualified, match="DACL"):
        qualify_acl()


def test_release_gate_cli_is_negative_now_and_positive_only_for_complete_evidence(
    tmp_path: Path,
) -> None:
    revision = "revision-1"
    command = [
        sys.executable,
        "-m",
        "deployment.platform_readiness",
        "--platform",
        "WINDOWS",
        "--current-revision",
        revision,
        "--expected-ci-provider",
        "test",
        "--expected-ci-run-id",
        "1",
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
        [
            *command,
            "--contract",
            str(contract_path),
            "--evidence",
            str(windows_path),
            "--evidence",
            str(core_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ready.returncode == 0
    assert "WINDOWS_PRODUCTION_READY=PASS" in ready.stdout


def _document(
    platform_name: str, runner_os: str, revision: str, results: list[dict[str, str]]
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "platform": platform_name,
        "source_revision": revision,
        "ci_provider": "test",
        "ci_run_id": "1",
        "runner_os": runner_os,
        "runner_arch": "x64",
        "generated_at_utc": "2026-01-01T00:00:00Z",
        "results": results,
    }


def _result(item: str, evidence_class: str, status: str = "PASS") -> dict[str, str]:
    return {
        "item": item,
        "status": status,
        "evidence_class": evidence_class,
        "test_or_probe": "test probe",
        "details": "executed",
    }


def _complete_windows_evidence(
    data: dict[str, object], revision: str
) -> tuple[dict[str, object], dict[str, object]]:
    acceptance = data["acceptance"]
    required = data["release_gates"]["WINDOWS_PRODUCTION_READY"]
    live = [
        _result(item, "LIVE_WINDOWS_INTEGRATION")
        for item in required
        if acceptance[item]["evidence_class"] == "LIVE_WINDOWS_INTEGRATION"
    ]
    core = [_result("CORE_REQUIRED_SUITES", "CROSS_OS_CI_MATRIX")]
    windows = _document("WINDOWS", "Windows", revision, live)
    windows["production_trust"] = _production_trust()
    return windows, _document("CROSS_PLATFORM_CORE", "Windows", revision, core)


def _production_trust() -> dict[str, str]:
    return {
        "ceremony_id": PRODUCTION_TRUST_CEREMONY_ID,
        "package_manifest_sha256": "a" * 64,
        "artifact_run_id": "37216551602",
        "frozen_production_trust": "PASS",
    }


def test_current_windows_evidence_is_accepted_only_when_complete() -> None:
    data = load_contract()
    windows, core = _complete_windows_evidence(data, "current")
    assert production_ready("WINDOWS", data, [windows, core], "current") is True
    windows["results"].pop()
    assert production_ready("WINDOWS", data, [windows, core], "current") is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document.pop("production_trust"),
        lambda document: document["production_trust"].update(extra="forbidden"),
        lambda document: document["production_trust"].update(ceremony_id="0" * 64),
        lambda document: document["production_trust"].update(package_manifest_sha256="abc"),
        lambda document: document["production_trust"].update(package_manifest_sha256="A" * 64),
        lambda document: document["production_trust"].update(artifact_run_id="0"),
        lambda document: document["production_trust"].update(artifact_run_id="not-a-run"),
        lambda document: document["production_trust"].update(frozen_production_trust="FAIL"),
    ],
    ids=[
        "missing-production-trust",
        "extra-production-trust-key",
        "wrong-ceremony-id",
        "malformed-manifest-sha",
        "uppercase-manifest-sha",
        "zero-artifact-run-id",
        "nonnumeric-artifact-run-id",
        "failed-frozen-production-trust",
    ],
)
def test_clean_install_production_trust_fails_closed(
    mutation: Callable[[dict[str, object]], object],
) -> None:
    data = load_contract()
    windows, core = _complete_windows_evidence(data, "current")
    mutation(windows)
    assert production_ready("WINDOWS", data, [windows, core], "current") is False


def test_production_trust_is_forbidden_without_clean_install_result() -> None:
    data = load_contract()
    evidence = _document(
        "WINDOWS",
        "Windows",
        "current",
        [_result("WINDOWS_SERVICE_INSTALLATION", "LIVE_WINDOWS_INTEGRATION")],
    )
    evidence["production_trust"] = _production_trust()
    blockers = blocking_items("WINDOWS", data, [evidence], "current")
    assert blockers == ["INVALID_EVIDENCE:malformed evidence document"]


def test_current_scm_slice_evidence_satisfies_only_its_exact_items() -> None:
    data = load_contract()
    scm_items = {
        "WINDOWS_SERVICE_INSTALLATION",
        "WINDOWS_SERVICE_START",
        "WINDOWS_GRACEFUL_STOP",
        "WINDOWS_MANUAL_RESTART",
    }
    windows = _document(
        "WINDOWS",
        "Windows",
        "current",
        [_result(item, "LIVE_WINDOWS_INTEGRATION") for item in sorted(scm_items)],
    )
    core = _document(
        "CROSS_PLATFORM_CORE",
        "Windows",
        "current",
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
    for name in (
        "FreshnessAuthority_implemented",
        "ROOT_PROOF_ISSUER_IMPLEMENTED",
        "production_substrate_implemented",
        "PRODUCTION_LOCAL_RUNTIME_AVAILABLE",
    ):
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
        (
            Path(__file__).resolve().parents[2] / ".github/workflows/platform-deployment.yml"
        ).read_text(encoding="utf-8")
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
        "core-linux",
        "core-windows",
        "core-macos",
        "windows-deployment-contract",
        "windows-deployment-integration",
        "windows-clean-install-integration",
        "windows-stage10-prepare-and-schedule-reboot",
        "windows-stage10-reboot-barrier",
        "windows-stage10-qualify-after-reboot",
    }
    assert "linux-deployment-integration" not in release["needs"]
    assert "macos-deployment-integration" not in release["needs"]
    release_commands = [step.get("run", "") for step in release["steps"]]
    assert any(
        "python -m deployment.platform_readiness --release-profile WINDOWS_BETA" in command
        and "--evidence evidence/core-matrix.json" in command
        and "--evidence evidence/windows-scm-evidence.json" in command
        and "--evidence evidence/windows-clean-install-evidence.json" in command
        for command in release_commands
    )
    integration_commands = [
        step.get("run", "") for step in jobs["windows-deployment-integration"]["steps"]
    ]
    assert any(
        "python -m deployment.windows_acceptance --mode github" in command
        for command in integration_commands
    )
    acceptance_step = next(
        step
        for step in jobs["windows-deployment-integration"]["steps"]
        if "python -m deployment.windows_acceptance --mode github" in step.get("run", "")
    )
    assert acceptance_step["timeout-minutes"] == 30
    parser_index = next(
        index
        for index, command in enumerate(integration_commands)
        if "System.Management.Automation.Language.Parser" in command
    )
    acceptance_index = next(
        index
        for index, command in enumerate(integration_commands)
        if "python -m deployment.windows_acceptance --mode github" in command
    )
    assert "deployment/windows_scm_probe.ps1" in integration_commands[parser_index]
    assert "$parseErrors.Count -ne 0" in integration_commands[parser_index]
    assert parser_index < acceptance_index
    assert not any(command.lstrip().startswith("echo ") for command in integration_commands)
    assert any(
        step.get("uses", "").startswith("actions/upload-artifact")
        for step in jobs["windows-deployment-integration"]["steps"]
    )


def test_stage10_workflow_uses_real_reboot_soak_and_evidence_producer() -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load(
        (root / ".github/workflows/platform-deployment.yml").read_text(encoding="utf-8")
    )
    jobs = workflow["jobs"]
    dispatch = workflow[True]["workflow_dispatch"]
    assert dispatch["inputs"]["run_stage10"] == {
        "description": "Run destructive/reboot Stage-10 lifecycle qualification",
        "required": True,
        "type": "boolean",
        "default": False,
    }
    opt_in = "github.event_name == 'workflow_dispatch' && inputs.run_stage10 == true"
    for job_name in (
        "windows-stage10-prepare-and-schedule-reboot",
        "windows-stage10-reboot-barrier",
        "windows-stage10-qualify-after-reboot",
    ):
        assert jobs[job_name]["if"] == opt_in
    assert jobs["windows-release-gate"]["if"] == (
        "always() && (github.event_name == 'pull_request' || inputs.run_stage10 == true)"
    )
    for job_name in ("linux-deployment-integration", "macos-deployment-integration"):
        assert jobs[job_name]["if"] == (
            "github.event_name == 'pull_request' || inputs.run_stage10 == true"
        )
    assert jobs["windows-stage10-prepare-and-schedule-reboot"]["runs-on"] == [
        "self-hosted",
        "Windows",
        "X64",
        "cryptohunter-stage10",
    ]
    prepare = "\n".join(
        step.get("run", "") for step in jobs["windows-stage10-prepare-and-schedule-reboot"]["steps"]
    )
    prepare_steps = jobs["windows-stage10-prepare-and-schedule-reboot"]["steps"]
    download_index = next(
        index
        for index, step in enumerate(prepare_steps)
        if step.get("with", {}).get("name") == "windows-clean-install-evidence"
    )
    prepare_index = next(
        index
        for index, step in enumerate(prepare_steps)
        if "windows_stage10_live.ps1 -Mode prepare-reboot" in step.get("run", "")
    )
    assert download_index < prepare_index
    assert "-QualifiedManifest stage10-qualified/installer-manifest.json" in prepare
    assert "-CleanInstallReceipt stage10-qualified/windows-clean-install-receipt.json" in prepare
    qualify = "\n".join(
        step.get("run", "") for step in jobs["windows-stage10-qualify-after-reboot"]["steps"]
    )
    assert "windows_stage10_live.ps1 -Mode prepare-reboot" in prepare
    assert "shutdown.exe /r" in prepare
    assert "windows_stage10_live.ps1 -Mode qualify" in qualify
    assert "platform_evidence windows-stage10" in qualify
    assert jobs["windows-stage10-qualify-after-reboot"]["timeout-minutes"] >= 24 * 60
    assert jobs["windows-stage10-reboot-barrier"]["runs-on"] == "ubuntu-latest"
    assert (
        jobs["windows-stage10-reboot-barrier"]["needs"]
        == "windows-stage10-prepare-and-schedule-reboot"
    )
    assert jobs["windows-stage10-qualify-after-reboot"]["needs"] == "windows-stage10-reboot-barrier"
    barrier = "\n".join(
        step.get("run", "") for step in jobs["windows-stage10-reboot-barrier"]["steps"]
    )
    assert "sleep 240" in barrier
    assert "shutdown.exe /r /t 75" in prepare
    probe = (root / "deployment/windows_stage10_live.ps1").read_text(encoding="utf-8")
    assert "Start-Sleep" in probe and "$SoakHours -lt 24" in probe
    assert "System.Diagnostics.Stopwatch" in probe
    assert "backend-readiness.json" in probe
    assert "runtime_provenance = $provenance" in probe
    assert probe.count("Service-Snapshot $runtimeProvenance.installed_backend_sha256") >= 3
    assert "missing or stale runtime provenance" in probe
    assert "installed backend executable changed" in probe
    assert "Get-AuthenticodeSignature" not in probe and "msiexec.exe" not in probe
    workflow_text = (root / ".github/workflows/platform-deployment.yml").read_text()
    assert "signtool" not in workflow_text
    diagnostic = next(
        step
        for step in jobs["windows-stage10-qualify-after-reboot"]["steps"]
        if step.get("name") == "Preserve Stage-10 lifecycle diagnostics"
    )
    assert diagnostic["if"] == "always()"
    assert "windows-stage10-transcript.log" in diagnostic["with"]["path"]
    download = next(
        step
        for step in jobs["windows-release-gate"]["steps"]
        if step.get("with", {}).get("name") == "windows-stage10-lifecycle-evidence"
    )
    assert download["if"] == opt_in
    release_commands = [step.get("run", "") for step in jobs["windows-release-gate"]["steps"]]
    assert any(
        "--evidence evidence/windows-stage10-evidence.json" in command
        for command in release_commands
    )


def test_v2_policy_separates_scope_compatibility_and_qualification() -> None:
    from deployment.platform_readiness import RELEASE_PROFILES, release_ready

    data = load_contract()
    assert data["schema_version"] == 2
    assert data["release_priority"] == {
        "WINDOWS": "PRIMARY",
        "LINUX": "SUPPORTED",
        "MACOS": "DEFERRED",
    }
    assert data["compatibility_platforms"] == ["WINDOWS", "LINUX", "MACOS"]
    assert data["release_profiles"] == RELEASE_PROFILES
    assert all(not production_ready(platform, data) for platform in data["platforms"])
    assert all(not release_ready(profile, data) for profile in RELEASE_PROFILES)


def _complete_linux_evidence(data: dict[str, object], revision: str) -> dict[str, object]:
    live = [
        _result(item, "LIVE_LINUX_INTEGRATION")
        for item in data["release_gates"]["LINUX_PRODUCTION_READY"]
        if data["acceptance"][item]["evidence_class"] == "LIVE_LINUX_INTEGRATION"
    ]
    return _document("LINUX", "Linux", revision, live)


def test_windows_only_beta_requires_windows_but_no_fabricated_linux_pass() -> None:
    from deployment.platform_readiness import release_blocking_items, release_ready

    data = load_contract()
    windows, core = _complete_windows_evidence(data, "current")
    evidence = [windows, core]
    assert production_ready("LINUX", data, evidence, "current") is False
    assert release_ready("WINDOWS_BETA", data, evidence, "current") is True
    for profile in ("WINDOWS_LINUX_BETA", "CRYPTOHUNTER_V1"):
        assert release_ready(profile, data, evidence, "current") is False
        assert all(
            item.startswith("LINUX:")
            for item in release_blocking_items(profile, data, evidence, "current")
        )
    # Scope is eligibility only: no acceptance status or legacy authority is rewritten.
    assert data == load_contract()


@pytest.mark.parametrize("profile", ["WINDOWS_BETA", "WINDOWS_LINUX_BETA", "CRYPTOHUNTER_V1"])
@pytest.mark.parametrize(
    "status", ["FAIL", "NOT_TESTED", "NOT_IMPLEMENTED", "NOT_APPLICABLE", "NOT_READY"]
)
def test_windows_not_ready_blocks_every_profile(profile: str, status: str) -> None:
    from deployment.platform_readiness import release_ready

    data = load_contract()
    windows, core = _complete_windows_evidence(data, "current")
    windows["results"][0]["status"] = status
    linux = _complete_linux_evidence(data, "current")
    assert release_ready(profile, data, [windows, core, linux], "current") is False


@pytest.mark.parametrize("profile", ["WINDOWS_LINUX_BETA", "CRYPTOHUNTER_V1"])
def test_linux_release_claim_requires_its_own_current_native_qualification(profile: str) -> None:
    from deployment.platform_readiness import release_ready

    data = load_contract()
    windows, core = _complete_windows_evidence(data, "current")
    linux = _complete_linux_evidence(data, "current")
    assert release_ready(profile, data, [windows, core, linux], "current") is True
    for mutation in (
        lambda document: document["results"].pop(),
        lambda document: document["results"][0].update(status="FAIL"),
        lambda document: document["results"][0].update(status="NOT_READY"),
        lambda document: document.update(source_revision="stale"),
        lambda document: document.update(runner_os="Windows"),
    ):
        altered = copy.deepcopy(linux)
        mutation(altered)
        assert release_ready(profile, data, [windows, core, altered], "current") is False


def test_macos_deferred_never_becomes_production_ready_even_with_synthetic_pass() -> None:
    data = load_contract()
    macos = _document(
        "MACOS",
        "macOS",
        "current",
        [
            _result(item, "LIVE_MACOS_INTEGRATION")
            for item in data["release_gates"]["MACOS_PRODUCTION_READY"]
            if item != "CORE_REQUIRED_SUITES"
        ],
    )
    _, core = _complete_windows_evidence(data, "current")
    assert production_ready("MACOS", data, [macos, core], "current") is False
    assert blocking_items("MACOS", data, [macos, core], "current") == ["PLATFORM_DEFERRED"]


def test_macos_compatibility_regression_keeps_shared_core_release_gate_closed() -> None:
    from deployment.platform_readiness import release_ready

    data = load_contract()
    windows, core = _complete_windows_evidence(data, "current")
    # CORE_REQUIRED_SUITES still aggregates all three OSes, including deferred macOS.
    core["results"][0]["status"] = "FAIL"
    assert not release_ready("WINDOWS_BETA", data, [windows, core], "current")
    root = Path(__file__).resolve().parents[2]
    plan = json.loads((root / "deployment/core_required_suites_v1.json").read_text())
    assert plan["required_platforms"] == ["Linux", "Windows", "macOS"]
    assert all(
        plan["acceptance_policy"][name] == 0
        for name in ("failures", "errors", "skips", "xfails", "unexpected_passes")
    )
    jobs = yaml.safe_load((root / ".github/workflows/platform-deployment.yml").read_text())["jobs"]
    assert jobs["core-macos"]["steps"] == jobs["core-linux"]["steps"]
    assert "core-macos" in jobs["windows-release-gate"]["needs"]
    compatibility = jobs["macos-deployment-integration"]
    assert compatibility["name"] == "macOS deployment compatibility (production deferred)"
    assert any("test_core_test_plan.py" in step.get("run", "") for step in compatibility["steps"])
    assert not any("continue-on-error" in step for step in compatibility["steps"])
    assert "continue-on-error" not in compatibility


@pytest.mark.parametrize("platform_name", ["UNKNOWN", "Darwin", "", None, [], True])
def test_unknown_platform_is_rejected_without_default_admission(platform_name: object) -> None:
    assert production_ready(platform_name, load_contract()) is False
    assert blocking_items(platform_name, load_contract()) == ["UNKNOWN_PLATFORM"]


@pytest.mark.parametrize("profile", ["UNKNOWN", "MACOS_BETA", "", None, [], True])
def test_unknown_release_profile_is_rejected(profile: object) -> None:
    from deployment.platform_readiness import release_blocking_items, release_ready

    assert release_ready(profile, load_contract()) is False
    assert release_blocking_items(profile, load_contract()) == ["UNKNOWN_RELEASE_PROFILE"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.pop("release_profiles"),
        lambda data: data.pop("compatibility_platforms"),
        lambda data: data.update(schema_version=1),
        lambda data: data.update(schema_version=True),
        lambda data: data.update(unreviewed=True),
        lambda data: data["release_priority"].update(MACOS="SUPPORTED"),
        lambda data: data["release_priority"].update(LINUX="UNKNOWN"),
        lambda data: data["release_priority"].pop("WINDOWS"),
        lambda data: data["release_profiles"].update(WINDOWS_BETA=[]),
        lambda data: data["release_profiles"].update(CRYPTOHUNTER_V1=["WINDOWS"]),
        lambda data: data["release_profiles"].update(MACOS_BETA=["MACOS"]),
        lambda data: data["compatibility_platforms"].remove("MACOS"),
        lambda data: data["semantic_invariants"].clear(),
        lambda data: data["matrix"]["core_tests"].update(MACOS="OPTIONAL"),
        lambda data: data["allowed_statuses"].append("GO"),
        lambda data: data["allowed_statuses"].append([]),
        lambda data: data["release_gates"]["WINDOWS_PRODUCTION_READY"].clear(),
        lambda data: data["release_gates"]["LINUX_PRODUCTION_READY"].pop(),
        lambda data: data["release_gates"]["MACOS_PRODUCTION_READY"].pop(),
        lambda data: data["acceptance"].pop("WINDOWS_ACL_QUALIFICATION"),
        lambda data: data["acceptance"]["CORE_REQUIRED_SUITES"].update(status="GO"),
        lambda data: data["acceptance"]["CORE_REQUIRED_SUITES"].update(
            evidence_class="STATIC_EXECUTABLE_CONTRACT", status="PASS"
        ),
        lambda data: data["acceptance"]["WINDOWS_ACL_QUALIFICATION"].update(
            evidence_class="STATIC_EXECUTABLE_CONTRACT", status="PASS"
        ),
        lambda data: data["mechanisms"].pop("MACOS"),
        lambda data: data["mechanisms"]["LINUX"].update(unknown="field"),
        lambda data: data["legacy_flags"].update(ROOT_PROOF_ISSUER_IMPLEMENTED=True),
    ],
)
def test_incomplete_unknown_or_weakened_policy_fails_closed(
    mutation: Callable[[dict[str, object]], object],
) -> None:
    from deployment.platform_readiness import (
        PolicyValidationError,
        release_ready,
        validate_contract,
    )

    data = load_contract()
    mutation(data)
    with pytest.raises(PolicyValidationError):
        validate_contract(data)
    assert blocking_items("WINDOWS", data) == ["INVALID_POLICY"]
    assert not production_ready("WINDOWS", data)
    assert not release_ready("WINDOWS_BETA", data)


def test_historical_v1_has_no_implicit_v2_scope_or_qualification(tmp_path: Path) -> None:
    from deployment.platform_readiness import PolicyValidationError

    historical = load_contract()
    historical["schema_version"] = 1
    historical["release_priority"]["MACOS"] = "SUPPORTED"
    historical.pop("compatibility_platforms")
    historical.pop("release_profiles")
    retained = copy.deepcopy(historical["acceptance"])
    path = tmp_path / "historical-v1.json"
    path.write_text(json.dumps(historical), encoding="utf-8")
    with pytest.raises(PolicyValidationError, match="UNSUPPORTED_POLICY_VERSION"):
        load_contract(path)
    assert json.loads(path.read_text())["acceptance"] == retained
    assert retained["MACOS_LAUNCHD_INTEGRATION"]["status"] == "NOT_IMPLEMENTED"
    assert not production_ready("MACOS", historical)


def test_release_profile_cli_and_incomplete_policy_are_fail_closed(tmp_path: Path) -> None:
    from deployment.platform_readiness import main

    data = load_contract()
    windows, core = _complete_windows_evidence(data, "current")
    paths = []
    for name, document in (("windows", windows), ("core", core)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        paths.extend(["--evidence", str(path)])
    common = [
        "--current-revision",
        "current",
        "--expected-ci-provider",
        "test",
        "--expected-ci-run-id",
        "1",
        *paths,
    ]
    assert main(["--release-profile", "WINDOWS_BETA", *common]) == 0
    assert main(["--release-profile", "CRYPTOHUNTER_V1", *common]) == 1
    assert main(["--platform", "MACOS", *common]) == 1
    bad_policy = tmp_path / "bad-policy.json"
    bad_policy.write_text("{}", encoding="utf-8")
    assert main(["--release-profile", "WINDOWS_BETA", "--contract", str(bad_policy), *common]) == 1


def test_duplicate_policy_version_is_not_reinterpreted_as_v2(tmp_path: Path) -> None:
    from deployment.platform_readiness import PolicyValidationError

    policy = tmp_path / "ambiguous-policy.json"
    raw = json.dumps(load_contract())
    policy.write_text(
        raw.replace('"schema_version": 2', '"schema_version": 1, "schema_version": 2')
    )
    with pytest.raises(PolicyValidationError, match="DUPLICATE_POLICY_FIELD"):
        load_contract(policy)


def test_historical_v1_macos_evidence_retains_its_class_without_qualifying_release() -> None:
    from deployment.platform_readiness import _validated_results

    data = load_contract()
    result = _result("MACOS_LAUNCHD_INTEGRATION", "LIVE_MACOS_INTEGRATION")
    document = _document("MACOS", "macOS", "historical-revision", [result])
    retained = copy.deepcopy(document)
    accepted = _validated_results(data, [document], "historical-revision", "test", "1")
    assert accepted["MACOS_LAUNCHD_INTEGRATION"] == result
    assert document == retained
    assert document["schema_version"] == 1
    assert not production_ready("MACOS", data, [document], "historical-revision")


def test_boolean_evidence_version_is_not_legacy_v1() -> None:
    data = load_contract()
    windows, core = _complete_windows_evidence(data, "current")
    windows["schema_version"] = True
    assert not production_ready("WINDOWS", data, [windows, core], "current")
