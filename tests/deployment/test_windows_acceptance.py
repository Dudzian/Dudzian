from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import deployment.windows_acceptance as windows_acceptance
from deployment.windows_strict_service_create import (
    StrictCreateFailure,
    StrictCreateResult,
    strict_create_service,
)
from deployment.windows_service_ownership import (
    INTENT_CREATED,
    SERVICE_PROVEN,
    OwnershipQualificationError,
    ServiceObservation,
    cleanup_policy,
    health_marker_cleanup_allowed,
    qualify_service_creation,
    service_cleanup_allowed,
)
from deployment.windows_service_recovery import (
    RESET_PERIOD_SECONDS,
    RecoveryQualificationError,
    configure_recovery,
)
from deployment.platform_readiness import blocking_items, load_contract
from deployment.windows_acceptance import (
    GITHUB_PROVIDER,
    LOCAL_PROVIDER,
    AcceptanceBoundary,
    WindowsAcceptanceError,
    run_acceptance,
)


class ReviewedBoundary(AcceptanceBoundary):
    def __init__(
        self,
        tmp_path: Path,
        *,
        host: str = "Windows",
        elevated: bool = True,
        dependency: str | None = None,
        core_failure: bool = False,
        scm_failure: str | None = None,
    ) -> None:
        self.tmp_path = tmp_path
        self.host = host
        self.elevated = elevated
        self.dependency = dependency
        self.core_failure = core_failure
        self.scm_failure = scm_failure

    def host_os(self) -> str:
        return self.host

    def is_elevated(self) -> bool:
        return self.elevated

    def staging_parent(self) -> Path:
        return self.tmp_path / "staging"

    def dependency_error(self, staging_parent: Path) -> str | None:
        return self.dependency

    def run_core(self, revision: str, provider: str, run_id: str, output: Path) -> None:
        if self.core_failure:
            raise RuntimeError("core failed")
        output.write_text(json.dumps({"provider": provider}), encoding="utf-8")

    def run_scm(self) -> dict[str, str]:
        if self.scm_failure:
            raise WindowsAcceptanceError("reviewed failure", self.scm_failure)
        return {
            "WINDOWS_NATIVE_PATH_INTEGRATION": "PASS",
            "WINDOWS_PROTECTED_CONFIGURATION": "PASS",
            "WINDOWS_PROTECTED_STATE_PATHS": "PASS",
            "WINDOWS_ACL_QUALIFICATION": "PASS",
            "WINDOWS_SERVICE_INSTALLATION": "PASS",
            "WINDOWS_AUTOSTART": "PASS",
            "WINDOWS_SERVICE_START": "PASS",
            "WINDOWS_GRACEFUL_STOP": "PASS",
            "WINDOWS_MANUAL_RESTART": "PASS",
            "WINDOWS_AUTOMATIC_CRASH_RESTART": "PASS",
            "cleanup": "PASS",
            "details": "reviewed boundary completed",
        }


def invoke(
    tmp_path: Path, boundary: ReviewedBoundary, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    monkeypatch.setenv("GITHUB_SERVER_URL", GITHUB_PROVIDER)
    monkeypatch.setenv("GITHUB_RUN_ID", "run-1")
    monkeypatch.setenv("GITHUB_SHA", "revision-1")
    core = tmp_path / "core.json"
    evidence = tmp_path / "scm.json"
    run_acceptance(
        mode="github",
        revision="revision-1",
        core_output=core,
        evidence_output=evidence,
        boundary=boundary,
    )
    return core, evidence


def test_non_windows_host_rejected_without_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    boundary = ReviewedBoundary(tmp_path, host="Linux")
    with pytest.raises(WindowsAcceptanceError, match="UNVERIFIED_ENVIRONMENT_LIMITATION"):
        invoke(tmp_path, boundary, monkeypatch)
    assert not list(tmp_path.glob("*.json"))


def test_windows_non_elevated_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    boundary = ReviewedBoundary(tmp_path, elevated=False)
    with pytest.raises(WindowsAcceptanceError, match="REQUIRES_ELEVATION"):
        invoke(tmp_path, boundary, monkeypatch)


@pytest.mark.parametrize(
    "reason", ["pywin32 is required", "project dependency check failed: broken"]
)
def test_missing_dependency_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
) -> None:
    with pytest.raises(WindowsAcceptanceError, match="DEPENDENCY_MISSING"):
        invoke(tmp_path, ReviewedBoundary(tmp_path, dependency=reason), monkeypatch)


def test_failed_core_plan_publishes_no_pass_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(WindowsAcceptanceError) as failure:
        invoke(tmp_path, ReviewedBoundary(tmp_path, core_failure=True), monkeypatch)
    assert failure.value.result == "WINDOWS_CORE_PLAN"
    assert not (tmp_path / "core.json").exists()
    assert not (tmp_path / "scm.json").exists()


@pytest.mark.parametrize(
    "item",
    [
        "WINDOWS_SERVICE_INSTALLATION",
        "WINDOWS_AUTOSTART",
        "WINDOWS_SERVICE_START",
        "WINDOWS_GRACEFUL_STOP",
        "WINDOWS_MANUAL_RESTART",
        "WINDOWS_AUTOMATIC_CRASH_RESTART",
        "WINDOWS_CLEANUP",
    ],
)
def test_each_scm_or_cleanup_failure_publishes_no_pass_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    item: str,
) -> None:
    with pytest.raises(WindowsAcceptanceError) as failure:
        invoke(tmp_path, ReviewedBoundary(tmp_path, scm_failure=item), monkeypatch)
    assert failure.value.result == item
    assert not (tmp_path / "core.json").exists()
    assert not (tmp_path / "scm.json").exists()


def test_successful_orchestration_uses_reviewed_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core, evidence_path = invoke(tmp_path, ReviewedBoundary(tmp_path), monkeypatch)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert core.exists()
    assert evidence["ci_provider"] == GITHUB_PROVIDER
    assert {result["item"] for result in evidence["results"]} == {
        "WINDOWS_NATIVE_PATH_INTEGRATION",
        "WINDOWS_PROTECTED_CONFIGURATION",
        "WINDOWS_PROTECTED_STATE_PATHS",
        "WINDOWS_ACL_QUALIFICATION",
        "WINDOWS_SERVICE_INSTALLATION",
        "WINDOWS_AUTOSTART",
        "WINDOWS_SERVICE_START",
        "WINDOWS_GRACEFUL_STOP",
        "WINDOWS_MANUAL_RESTART",
        "WINDOWS_AUTOMATIC_CRASH_RESTART",
    }
    assert all(result["status"] == "PASS" for result in evidence["results"])


def test_local_identity_cannot_satisfy_github_gate_but_github_scm_items_can(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(windows_acceptance, "_revision", lambda value: value or "revision-1")
    boundary = ReviewedBoundary(tmp_path)
    core = tmp_path / "local-core.json"
    local_path = tmp_path / "local-scm.json"
    run_acceptance(
        mode="local",
        revision="revision-1",
        core_output=core,
        evidence_output=local_path,
        boundary=boundary,
    )
    local = json.loads(local_path.read_text(encoding="utf-8"))
    assert local["ci_provider"] == LOCAL_PROVIDER
    data = load_contract()
    local_blockers = blocking_items(
        "WINDOWS",
        data,
        [local],
        "revision-1",
        GITHUB_PROVIDER,
        "run-1",
    )
    assert local_blockers[0].startswith("INVALID_EVIDENCE:untrusted CI provider")

    _, github_path = invoke(tmp_path, boundary, monkeypatch)
    github = json.loads(github_path.read_text(encoding="utf-8"))
    blockers = blocking_items(
        "WINDOWS",
        data,
        [github],
        "revision-1",
        GITHUB_PROVIDER,
        "run-1",
    )
    for item in (
        "WINDOWS_SERVICE_INSTALLATION",
        "WINDOWS_SERVICE_START",
        "WINDOWS_GRACEFUL_STOP",
        "WINDOWS_MANUAL_RESTART",
    ):
        assert item not in blockers
    assert "WINDOWS_CLEAN_INSTALL" in blockers


def test_probe_installs_service_before_first_service_identity_acl() -> None:
    probe = (Path(__file__).resolve().parents[2] / "deployment/windows_scm_probe.ps1").read_text(
        encoding="utf-8"
    )
    install = probe.index(
        "& $PythonExecutable $harness acceptance-install-strict --username $identity"
    )
    query = probe.index("Get-CimInstance Win32_Service -Filter \"Name='$service'\"", install)
    failed_exit = probe.index("if ($strictCreateExitCode -ne 0)", install)
    proven = probe.index('$ownership.ownership_phase -cne "SERVICE_PROVEN"', query)
    installed_by_probe = probe.index("$serviceInstalledByProbe = $true", proven)
    first_grant = probe.index("/grant", query)
    start = probe.index("Start-Service $service", first_grant)
    assert install < query < first_grant < start
    assert install < failed_exit < query < proven < installed_by_probe < first_grant
    assert "NT SERVICE\\CryptoHunterBackend" in probe
    assert "LocalSystem" not in probe
    assert "NetworkService" not in probe
    assert "Everyone" not in probe


def test_strict_create_configures_autostart_without_update_fallback() -> None:
    harness = (
        Path(__file__).resolve().parents[2] / "deployment/windows_test_service.py"
    ).read_text(encoding="utf-8")
    strict = (
        Path(__file__).resolve().parents[2] / "deployment/windows_strict_service_create.py"
    ).read_text(encoding="utf-8")
    assert "startType=win32service.SERVICE_AUTO_START" in harness
    assert "SERVICE_DEMAND_START" not in harness
    assert "UpdateService" not in harness
    assert "UpdateService" not in strict
    assert "ERROR_SERVICE_EXISTS" in strict


def test_probe_ownership_record_is_created_after_absence_check_before_install() -> None:
    probe = (Path(__file__).resolve().parents[2] / "deployment/windows_scm_probe.ps1").read_text(
        encoding="utf-8"
    )
    absence = probe.index("if (Get-Service $service -ErrorAction SilentlyContinue)")
    create_record = probe.index("[System.IO.File]::Open(", absence)
    install = probe.index("& $PythonExecutable $harness acceptance-install-strict", create_record)
    assert absence < create_record < install
    for field in (
        "run_token",
        "service_name",
        "service_identity",
        "python_executable",
        "harness",
    ):
        assert field in probe[create_record - 500 : install]


def test_cleanup_only_same_service_name_without_matching_record_is_never_owned() -> None:
    probe = (Path(__file__).resolve().parents[2] / "deployment/windows_scm_probe.ps1").read_text(
        encoding="utf-8"
    )
    cleanup_start = probe.index("if ($CleanupOnly)")
    cleanup = probe[
        cleanup_start : probe.index(
            '$primaryFailure = "[CLEANUP] timeout recovery cleanup requested"', cleanup_start
        )
    ]
    assert "Test-BaseOwnershipRecord $ownership" in cleanup
    assert "current-run service ownership is not proven" in cleanup
    assert cleanup.index("Test-BaseOwnershipRecord") < cleanup.index(
        "$serviceOwnershipProven = $true"
    )
    assert "$configured" in cleanup
    assert '$ownership.ownership_phase -cne "SERVICE_PROVEN"' in cleanup
    assert '$ownership.strict_create_result -cne "CREATED"' in cleanup
    assert "$configured.Name -cne $service" in cleanup
    assert "$configured.StartName -cne $identity" in cleanup
    assert "-not $expectedPath" in cleanup
    assert "$configured.PathName -cne $expectedPath" in cleanup
    assert "-not $expectedHost" in cleanup
    assert "$actualHost -cne $expectedHost" in cleanup
    assert "-not $expectedSid" in cleanup
    assert "$actualSid -cne $expectedSid" in cleanup


def test_preexisting_or_token_mismatched_service_is_untouched_and_cleanup_fails() -> None:
    probe = (Path(__file__).resolve().parents[2] / "deployment/windows_scm_probe.ps1").read_text(
        encoding="utf-8"
    )
    ownership_test = probe[
        probe.index("function Test-BaseOwnershipRecord") : probe.index("function Add-PlannedGrant")
    ]
    assert "$Record.run_token -ceq $WindowsAcceptanceRunToken" in ownership_test
    finally_block = probe[probe.rindex("} finally {") : probe.index("$result | ConvertTo-Json")]
    ownership_guard = finally_block.index("if ($intentOwned)")
    stop = finally_block.index("Stop-Service $service")
    remove = finally_block.index("& $PythonExecutable $harness remove")
    assert ownership_guard < stop < remove
    assert "if ($cleanupFailures.Count -eq 0)" in finally_block


def test_matching_owned_cleanup_removes_run_resources_and_record_before_pass() -> None:
    probe = (Path(__file__).resolve().parents[2] / "deployment/windows_scm_probe.ps1").read_text(
        encoding="utf-8"
    )
    finally_block = probe[probe.rindex("} finally {") : probe.index("$result | ConvertTo-Json")]
    for operation in (
        "Stop-Service $service",
        "Remove-Item $marker",
        "Remove-ServiceGrant",
        "& $PythonExecutable $harness remove",
        "Remove-Item -LiteralPath $ownershipPath",
        '"ownership record remains"',
    ):
        assert operation in finally_block
    assert finally_block.index("Remove-ServiceGrant") < finally_block.index(
        "& $PythonExecutable $harness remove"
    )
    assert finally_block.index('"ownership record remains"') < finally_block.index(
        '$result.cleanup = "PASS"'
    )
    marker_guard = finally_block.index("if ($serviceOwnershipProven)")
    marker_remove = finally_block.index("Remove-Item $marker", marker_guard)
    marker_verify = finally_block.index("owned health marker remains", marker_remove)
    assert marker_guard < marker_remove < marker_verify


def test_unproven_cleanup_leaves_foreign_marker_and_reports_failure() -> None:
    probe = (Path(__file__).resolve().parents[2] / "deployment/windows_scm_probe.ps1").read_text(
        encoding="utf-8"
    )
    finally_block = probe[probe.rindex("} finally {") : probe.index("$result | ConvertTo-Json")]
    assert "unowned health marker exists after install attempt; left untouched" in finally_block
    assert "-not $serviceOwnershipProven -and (Test-Path -LiteralPath $marker)" in finally_block


def test_partial_acl_timeout_uses_only_recorded_planned_grants() -> None:
    probe = (Path(__file__).resolve().parents[2] / "deployment/windows_scm_probe.ps1").read_text(
        encoding="utf-8"
    )
    plan = probe[probe.index("function Add-PlannedGrant") : probe.index("try {")]
    assert (
        plan.index("Test-ServiceGrant")
        < plan.index("ownership.grants")
        < plan.index("Save-OwnershipRecord")
    )
    finally_block = probe[probe.rindex("} finally {") : probe.index("$result | ConvertTo-Json")]
    assert "foreach ($target in @($ownership.grants))" in finally_block
    assert "pre-existing service ACL prevents ownership-safe test grant" in probe
    assert "$script:serviceSid" in probe


def test_install_success_then_acl_failure_still_runs_owned_cleanup() -> None:
    probe = (Path(__file__).resolve().parents[2] / "deployment/windows_scm_probe.ps1").read_text(
        encoding="utf-8"
    )
    installed = probe.index("$serviceInstalledByProbe = $true")
    first_grant_failure = probe.index("test health directory ACL provisioning failed")
    finally_block = probe.index("} finally {", first_grant_failure)
    remove_service = probe.index("& $PythonExecutable $harness remove", finally_block)
    assert installed < first_grant_failure < finally_block < remove_service


def test_strict_create_success_calls_create_once_and_never_updates() -> None:
    calls = {"create": 0, "update": 0}

    def create() -> None:
        calls["create"] += 1

    result = strict_create_service(create)
    assert result == StrictCreateResult.CREATED
    assert calls == {"create": 1, "update": 0}


def test_error_service_exists_is_terminal_and_foreign_service_is_unchanged() -> None:
    class ServiceExistsError(RuntimeError):
        winerror = 1073

    foreign = {"start_name": "FOREIGN\\Account", "image_path": "foreign.exe"}
    before = foreign.copy()
    calls = {"create": 0, "change_config": 0}

    def create() -> None:
        calls["create"] += 1
        raise ServiceExistsError("already exists")

    with pytest.raises(StrictCreateFailure) as failure:
        strict_create_service(create)
    assert failure.value.result == StrictCreateResult.ALREADY_EXISTS
    assert calls == {"create": 1, "change_config": 0}
    assert foreign == before
    assert service_cleanup_allowed(_ownership_intent()) is False


def test_unknown_strict_create_failure_is_not_claimed_as_partial_create() -> None:
    def create() -> None:
        raise RuntimeError("failure timing is unknown")

    with pytest.raises(StrictCreateFailure) as failure:
        strict_create_service(create)
    assert failure.value.result == StrictCreateResult.FAILED
    assert service_cleanup_allowed(_ownership_intent()) is False


def test_acceptance_strict_create_path_does_not_use_handle_command_line_install() -> None:
    root = Path(__file__).resolve().parents[2]
    probe = (root / "deployment/windows_scm_probe.ps1").read_text(encoding="utf-8")
    harness = (root / "deployment/windows_test_service.py").read_text(encoding="utf-8")
    assert "acceptance-install-strict" in probe
    assert "--startup manual" not in probe
    strict_function = harness[harness.index("def acceptance_install_strict") :]
    assert "win32serviceutil.InstallService(" in strict_function
    assert "ChangeServiceConfig" not in strict_function
    assert "HandleCommandLine" not in strict_function.split('if __name__ == "__main__":')[0]


def test_probe_uses_windows_powershell_compatible_fail_closed_path_qualification() -> None:
    root = Path(__file__).resolve().parents[2]
    probe = (root / "deployment/windows_scm_probe.ps1").read_text(encoding="utf-8")
    helper = (root / "deployment/windows_path_qualification.ps1").read_text(encoding="utf-8")
    workflow = (root / ".github/workflows/platform-deployment.yml").read_text(encoding="utf-8")

    assert "IsPathFullyQualified" not in probe + helper
    assert "GetFullPath" not in probe + helper
    assert "Test-FullyQualifiedWindowsPath" in probe
    assert "shell: powershell" in workflow
    assert "windows_path_qualification.ps1" in workflow


def test_acl_reader_is_exact_native_module_and_pwsh_inheritance_is_proven() -> None:
    root = Path(__file__).resolve().parents[2]
    helper = (root / "deployment/windows_native_acl.ps1").read_text(encoding="utf-8")
    probe = (root / "deployment/windows_scm_probe.ps1").read_text(encoding="utf-8")
    regression = (root / "deployment/windows_native_acl_regression.ps1").read_text(encoding="utf-8")
    launcher = (root / "deployment/windows_native_acl_regression.py").read_text(encoding="utf-8")
    workflow = (root / ".github/workflows/platform-deployment.yml").read_text(encoding="utf-8")

    identity_function = re.search(
        r"function Test-WindowsPathIdentity.*?^}", helper, re.MULTILINE | re.DOTALL
    )
    assert identity_function is not None
    identity_contract = identity_function.group(0)
    assert identity_contract.count("[System.IO.Path]::GetFullPath") == 2
    assert identity_contract.count(".TrimEnd('\\')") == 2
    assert (
        "[StringComparer]::OrdinalIgnoreCase.Equals($actualRoot, $expectedRoot)"
        in identity_contract
    )
    assert "-ceq" not in identity_contract
    assert "-cne" not in identity_contract

    containment_function = re.search(
        r"function Test-WindowsPathWithinRoot.*?^}", helper, re.MULTILINE | re.DOTALL
    )
    assert containment_function is not None
    containment_contract = containment_function.group(0)
    assert containment_contract.count("[System.IO.Path]::GetFullPath") == 2
    assert ".TrimEnd([char[]]@('\\', '/'))" in containment_contract
    assert (
        "$rootWithSeparator = $fullRoot + [System.IO.Path]::DirectorySeparatorChar"
        in containment_contract
    )
    assert (
        "$fullPath.StartsWith($rootWithSeparator, [StringComparison]::OrdinalIgnoreCase)"
        in containment_contract
    )
    assert ".StartsWith($fullRoot" not in containment_contract

    manifest_assignment = (
        "$nativeSecurityManifest = Join-Path $PSHOME "
        '"Modules\\Microsoft.PowerShell.Security\\Microsoft.PowerShell.Security.psd1"'
    )
    assert manifest_assignment in helper
    assert (
        "Import-Module -Name $nativeSecurityManifest -Force -PassThru -ErrorAction Stop" in helper
    )
    assert "$nativeSecurityAssembly = Join-Path $PSHOME" not in helper
    assert "Test-Path -LiteralPath $nativeSecurityAssembly -PathType Leaf" not in helper
    assert '$nativeSecurityModule.Name -cne "Microsoft.PowerShell.Security"' in helper
    assert "Test-WindowsPathIdentity $nativeSecurityModule.Path $nativeSecurityManifest" in helper
    get_acl_pipeline = helper[helper.index("$nativeGetAcl = Get-Command") :]
    assert "Get-Command -Name Get-Acl -Module Microsoft.PowerShell.Security" in get_acl_pipeline
    assert (
        "$nativeGetAclAssemblyObject = $nativeGetAcl.ImplementingType.Assembly" in get_acl_pipeline
    )
    assert "$nativeGetAclAssembly = $nativeGetAclAssemblyObject.Location" in get_acl_pipeline
    assert "[System.Management.Automation.CommandTypes]::Cmdlet" in get_acl_pipeline
    assert '$nativeGetAcl.ModuleName -cne "Microsoft.PowerShell.Security"' in get_acl_pipeline
    assert (
        "Test-WindowsPathIdentity $nativeGetAcl.Module.Path $nativeSecurityManifest"
        in get_acl_pipeline
    )
    assert "$null -eq $nativeGetAcl.ImplementingType" in get_acl_pipeline
    assert (
        '$nativeGetAcl.ImplementingType.FullName -cne "Microsoft.PowerShell.Commands.GetAclCommand"'
        in get_acl_pipeline
    )
    assert "$null -eq $nativeGetAcl.ImplementingType.Assembly" in get_acl_pipeline
    expected_assembly = (
        '"Microsoft.PowerShell.Security, Version=3.0.0.0, Culture=neutral, '
        'PublicKeyToken=31bf3856ad364e35"'
    )
    assert f"$nativeGetAclAssemblyIdentity = {expected_assembly}" in get_acl_pipeline
    assert (
        "$nativeGetAclAssemblyObject.FullName -cne $nativeGetAclAssemblyIdentity"
        in get_acl_pipeline
    )
    assert 'PSObject.Properties["GlobalAssemblyCache"]' in get_acl_pipeline
    assert "$nativeGetAclAssemblyObject.GlobalAssemblyCache -ne $true" in get_acl_pipeline
    assert "Test-Path -LiteralPath $nativeGetAclAssembly -PathType Leaf" in get_acl_pipeline
    assert (
        "[System.IO.Path]::GetFileName($nativeGetAclAssembly) -cne "
        '"Microsoft.PowerShell.Security.dll"' in get_acl_pipeline
    )
    assert "$nativeWindowsRoot = $PSHOME" in get_acl_pipeline
    assert "$parentIndex -lt 3" in get_acl_pipeline
    assert "[System.IO.Directory]::GetParent($nativeWindowsRoot)" in get_acl_pipeline
    assert (
        "$nativeGacSecurityRoot = Join-Path $nativeWindowsRoot "
        '"Microsoft.Net\\assembly\\GAC_MSIL\\Microsoft.PowerShell.Security"' in get_acl_pipeline
    )
    assert (
        "Test-WindowsPathWithinRoot $nativeGetAclAssembly $nativeGacSecurityRoot"
        in get_acl_pipeline
    )
    assert "NATIVE_SECURITY_EXECUTABLE_AUTHORITY_LAYOUT_UNQUALIFIED" not in get_acl_pipeline
    assert "ModuleBase $nativeSecurityRoot" not in helper
    assert "$env:PSModulePath" not in helper
    assert "Get-Module -ListAvailable" not in helper
    assert "Test-NativePrincipalGrant $Target $identity $script:serviceSid" in probe
    assert "Get-Acl -LiteralPath $Target" not in probe
    assert '["powershell.exe", "-NoProfile"' in launcher
    assert 'WINDOWS_ACL_REGRESSION_PARENT_EDITION") != "Core"' in launcher
    assert 'payload.get("child_edition") != "Desktop"' in launcher
    assert "$PSVersionTable.PSEdition" in regression
    assert "$PSHOME" in regression
    assert "$env:PSModulePath" not in regression
    assert "security_module_path" in regression
    for diagnostic in (
        "SECURITY_IMPORT_NAME",
        "SECURITY_IMPORT_MODULE_BASE",
        "SECURITY_IMPORT_PATH",
        "GET_ACL_COMMAND_TYPE",
        "GET_ACL_MODULE_NAME",
        "GET_ACL_MODULE_BASE",
        "GET_ACL_MODULE_PATH",
        "GET_ACL_IMPLEMENTING_TYPE",
        "GET_ACL_ASSEMBLY_FULL_NAME",
        "GET_ACL_ASSEMBLY_LOCATION",
        "PS_EDITION",
        "PS_HOME",
    ):
        assert diagnostic in helper
    assert 'payload.get("get_acl_command_type") != "Cmdlet"' in launcher
    assert 'payload.get("get_acl_module_name") != "Microsoft.PowerShell.Security"' in launcher
    assert (
        'payload.get("get_acl_implementing_type") != '
        '"Microsoft.PowerShell.Commands.GetAclCommand"' in launcher
    )
    assert "PublicKeyToken=31bf3856ad364e35" in launcher
    assert 'not payload.get("get_acl_assembly_location")' in launcher
    assert 'payload.get("get_acl_assembly_file_exists") is not True' in launcher
    assert 'payload.get("get_acl_global_assembly_cache") is not True' in launcher
    assert 'payload.get("security_executable_authority_qualified") is not True' in launcher
    assert "get_acl_global_assembly_cache" in regression
    assert "native_gac_security_root" in regression
    assert "security_executable_authority_qualified = $true" in regression
    assert "before = $before" in regression
    assert "after_grant = $afterGrant" in regression
    assert "after_remove = $afterRemove" in regression
    assert "shell: pwsh" in workflow
    assert "python -m deployment.windows_native_acl_regression" in workflow


@pytest.mark.skipif(shutil.which("powershell.exe") is None, reason="requires Windows PowerShell")
def test_windows_path_identity_runtime_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    helper = root / "deployment/windows_native_acl.ps1"
    cases = (
        (r"C:\Windows\System32\Foo", r"C:\WINDOWS\system32\Foo", True),
        (r"C:\Path\Module", "C:\\Path\\Module\\", True),
        (r"C:\Windows\System32\Foo", r"D:\Windows\System32\Foo", False),
        (r"C:\Windows\System32\Foo", r"C:\Windows\SysWOW64\Foo", False),
    )
    assertions = "; ".join(
        "if ((Test-WindowsPathIdentity '%s' '%s') -ne $%s) { exit 1 }"
        % (actual, expected, str(wanted).lower())
        for actual, expected, wanted in cases
    )
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            ". '%s'; %s" % (helper, assertions),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _ownership_intent() -> dict[str, object]:
    return {
        "run_token": "a" * 64,
        "service_name": "CryptoHunterBackend",
        "service_identity": r"NT SERVICE\CryptoHunterBackend",
        "python_executable": r"C:\Python\python.exe",
        "harness": r"C:\repo\deployment\windows_test_service.py",
        "ownership_phase": INTENT_CREATED,
        "service_path_name": None,
        "service_host": None,
        "service_sid": None,
        "grants": [],
    }


def _canonical_observation() -> ServiceObservation:
    return ServiceObservation(
        name="CryptoHunterBackend",
        start_name=r"NT SERVICE\CryptoHunterBackend",
        path_name=r'"C:\Python\PythonService.exe"',
        service_host=r"C:\Python\PythonService.exe",
        service_sid="S-1-5-80-123",
    )


def test_failed_install_same_name_foreign_service_never_becomes_cleanup_owned() -> None:
    intent = _ownership_intent()
    with pytest.raises(OwnershipQualificationError, match="strict create did not prove"):
        qualify_service_creation(
            intent,
            run_token="a" * 64,
            python_executable=r"C:\Python\python.exe",
            harness=r"C:\repo\deployment\windows_test_service.py",
            strict_create_result="ALREADY_EXISTS",
            observation=_canonical_observation(),
        )
    assert intent["ownership_phase"] == INTENT_CREATED
    assert service_cleanup_allowed(intent) is False


def test_failed_install_foreign_service_and_marker_are_preserved_with_cleanup_failure() -> None:
    intent = _ownership_intent()
    policy = cleanup_policy(intent, service_exists=True, health_marker_exists=True)
    assert policy.remove_service is False
    assert policy.remove_health_marker is False
    assert policy.ownership_failure is True
    assert health_marker_cleanup_allowed(intent) is False


def test_intent_cleanup_only_preserves_foreign_marker_even_without_service() -> None:
    policy = cleanup_policy(
        _ownership_intent(),
        service_exists=False,
        health_marker_exists=True,
    )
    assert policy.remove_service is False
    assert policy.remove_health_marker is False
    assert policy.ownership_failure is True


def test_token_mismatch_cannot_authorize_service_or_marker_cleanup() -> None:
    intent = _ownership_intent()
    with pytest.raises(OwnershipQualificationError, match="does not match this run"):
        qualify_service_creation(
            intent,
            run_token="b" * 64,
            python_executable=r"C:\Python\python.exe",
            harness=r"C:\repo\deployment\windows_test_service.py",
            strict_create_result="CREATED",
            observation=_canonical_observation(),
        )
    policy = cleanup_policy(intent, service_exists=True, health_marker_exists=True)
    assert policy.remove_service is False
    assert policy.remove_health_marker is False
    assert policy.ownership_failure is True


def test_successful_install_exact_config_transitions_to_service_proven() -> None:
    intent = _ownership_intent()
    proven = qualify_service_creation(
        intent,
        run_token="a" * 64,
        python_executable=r"C:\Python\python.exe",
        harness=r"C:\repo\deployment\windows_test_service.py",
        strict_create_result="CREATED",
        observation=_canonical_observation(),
    )
    assert intent["ownership_phase"] == INTENT_CREATED
    assert proven["ownership_phase"] == SERVICE_PROVEN
    assert service_cleanup_allowed(proven) is True
    assert proven["service_path_name"] == _canonical_observation().path_name
    policy = cleanup_policy(proven, service_exists=True, health_marker_exists=True)
    assert policy.remove_service is True
    assert policy.remove_health_marker is True
    assert policy.ownership_failure is False
    assert health_marker_cleanup_allowed(proven) is True


@pytest.mark.parametrize("mismatch", ["start_name", "path_name", "service_host"])
def test_successful_install_config_mismatch_never_authorizes_cleanup(mismatch: str) -> None:
    intent = _ownership_intent()
    values = _canonical_observation().__dict__.copy()
    values[mismatch] = ""
    with pytest.raises(OwnershipQualificationError):
        qualify_service_creation(
            intent,
            run_token="a" * 64,
            python_executable=r"C:\Python\python.exe",
            harness=r"C:\repo\deployment\windows_test_service.py",
            strict_create_result="CREATED",
            observation=ServiceObservation(**values),
        )
    assert service_cleanup_allowed(intent) is False


def test_exact_sys_executable_is_propagated_even_when_path_python_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    payload = {
        "WINDOWS_NATIVE_PATH_INTEGRATION": "PASS",
        "WINDOWS_PROTECTED_CONFIGURATION": "PASS",
        "WINDOWS_PROTECTED_STATE_PATHS": "PASS",
        "WINDOWS_ACL_QUALIFICATION": "PASS",
        "WINDOWS_SERVICE_INSTALLATION": "PASS",
        "WINDOWS_AUTOSTART": "PASS",
        "WINDOWS_SERVICE_START": "PASS",
        "WINDOWS_GRACEFUL_STOP": "PASS",
        "WINDOWS_MANUAL_RESTART": "PASS",
        "WINDOWS_AUTOMATIC_CRASH_RESTART": "PASS",
        "cleanup": "PASS",
    }

    def completed(command: list[str], **kwargs: object) -> object:
        calls.append(command)
        return type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": json.dumps(payload),
                "stderr": "",
            },
        )()

    monkeypatch.setenv("PATH", str(Path("wrong-path-python")))
    monkeypatch.setattr(windows_acceptance.subprocess, "run", completed)
    AcceptanceBoundary().run_scm()
    assert calls[0][calls[0].index("-PythonExecutable") + 1] == windows_acceptance.sys.executable
    assert "python" not in calls[0]


def test_timeout_invokes_cleanup_only_with_same_exact_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def timeout_then_cleanup(command: list[str], **kwargs: object) -> object:
        calls.append(command)
        if len(calls) == 1:
            raise windows_acceptance.subprocess.TimeoutExpired(command, 180)
        return type(
            "Completed",
            (),
            {
                "returncode": 1,
                "stdout": json.dumps({"cleanup": "PASS"}),
                "stderr": "",
            },
        )()

    monkeypatch.setattr(windows_acceptance.subprocess, "run", timeout_then_cleanup)
    with pytest.raises(WindowsAcceptanceError, match="timed out"):
        AcceptanceBoundary().run_scm()
    assert calls[1][-1] == "-CleanupOnly"
    assert calls[0][calls[0].index("-PythonExecutable") + 1] == windows_acceptance.sys.executable
    assert calls[1][calls[1].index("-PythonExecutable") + 1] == windows_acceptance.sys.executable
    normal_token = calls[0][calls[0].index("-WindowsAcceptanceRunToken") + 1]
    cleanup_token = calls[1][calls[1].index("-WindowsAcceptanceRunToken") + 1]
    assert normal_token == cleanup_token
    assert len(normal_token) == 64
    int(normal_token, 16)


def test_second_artifact_publish_failure_rolls_back_first_new_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    moves = 0

    def failing_replace(source: Path, target: Path) -> None:
        nonlocal moves
        moves += 1
        if moves == 2:
            raise OSError("second publication failed")
        source.replace(target)

    def publisher(staged_core: Path, core: Path, staged_evidence: Path, evidence: Path) -> None:
        windows_acceptance.publish_artifacts(
            staged_core,
            core,
            staged_evidence,
            evidence,
            replace=failing_replace,
        )

    monkeypatch.setenv("GITHUB_SERVER_URL", GITHUB_PROVIDER)
    monkeypatch.setenv("GITHUB_RUN_ID", "run-1")
    monkeypatch.setenv("GITHUB_SHA", "revision-1")
    core = tmp_path / "core.json"
    evidence = tmp_path / "scm.json"
    with pytest.raises(WindowsAcceptanceError, match="PUBLICATION_FAILED"):
        run_acceptance(
            mode="github",
            revision="revision-1",
            core_output=core,
            evidence_output=evidence,
            boundary=ReviewedBoundary(tmp_path),
            publisher=publisher,
        )
    assert moves == 2
    assert not core.exists()
    assert not evidence.exists()


def test_existing_outputs_are_preserved_and_run_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_SERVER_URL", GITHUB_PROVIDER)
    monkeypatch.setenv("GITHUB_RUN_ID", "run-1")
    monkeypatch.setenv("GITHUB_SHA", "revision-1")
    core = tmp_path / "core.json"
    evidence = tmp_path / "scm.json"
    core.write_text("approved-core", encoding="utf-8")
    evidence.write_text("approved-scm", encoding="utf-8")
    with pytest.raises(WindowsAcceptanceError, match="OUTPUT_ALREADY_EXISTS"):
        run_acceptance(
            mode="github",
            revision="revision-1",
            core_output=core,
            evidence_output=evidence,
            boundary=ReviewedBoundary(tmp_path),
        )
    assert core.read_text(encoding="utf-8") == "approved-core"
    assert evidence.read_text(encoding="utf-8") == "approved-scm"
