from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import zipfile

import pytest

from deployment import windows_stage9_clean_install as clean_install
from deployment.windows_installer.build import (
    InstallerBuildError,
    canonical_wix_version,
    msi_version,
    normalize_postgresql_archive,
    verify_postgresql,
)
from deployment.windows_installer.contract import CONTRACT
from deployment.windows_installer.provision import commit, create_journal, rollback
from deployment.windows_installer.postgresql_service import (
    create_suspended_in_job,
    require_clean_exit,
    wait_ready,
)


ROOT = Path(__file__).parents[2]
WXS = (ROOT / "deployment/windows_installer/Product.wxs").read_text(encoding="utf-8")


def test_canonical_msi_identity_and_scope() -> None:
    assert 'Scope="perMachine"' in WXS
    assert 'Platform="x64"' not in WXS
    builder = (ROOT / "deployment/windows_installer/build.py").read_text()
    assert '"-arch"' in builder and '"x64"' in builder
    assert "ProgramFiles64Folder" in WXS
    assert 'Directory Id="INSTALLFOLDER" Name="CryptoHunter"' in WXS
    assert CONTRACT.wix_version.startswith("7.")


def test_exact_service_contract_and_native_removal() -> None:
    for name in (CONTRACT.backend_service, CONTRACT.verifier_service, CONTRACT.postgresql_service):
        assert f'Name="{name}"' in WXS
        assert f'Account="NT SERVICE\\{name}"' in WXS
        assert re.search(rf'<ServiceControl[^>]+Name="{name}"[^>]+Remove="uninstall"', WXS)
    assert (
        'Name="CryptoHunterBackend"\n             DisplayName="CryptoHunter Backend" Type="ownProcess" Start="auto"'
        in WXS
    )
    assert '<ServiceDependency Id="CryptoHunterPostgreSQL" />' in WXS
    assert "LocalSystem" not in WXS


def test_service_sid_exists_before_privileged_provisioning() -> None:
    assert '<Custom Action="ProvisionMachine" Before="StartServices"' in WXS
    assert '<Custom Action="RollbackProvisionMachine" After="InstallServices"' in WXS
    assert 'Execute="deferred" Impersonate="no" Return="check" HideTarget="yes"' in WXS


def test_provisioning_custom_actions_quote_trailing_directory_separators_safely() -> None:
    for action in ("ProvisionMachine", "RollbackProvisionMachine", "CommitProvisionMachine"):
        custom_action = re.search(rf'<CustomAction Id="{action}"[^>]+>', WXS)
        assert custom_action is not None
        definition = custom_action.group()
        assert "&quot;[INSTALLFOLDER].&quot;" in definition
        assert "&quot;[PROGRAMDATAROOT].&quot;" in definition
        assert "&quot;[INSTALLFOLDER]&quot;" not in definition
        assert "&quot;[PROGRAMDATAROOT]&quot;" not in definition


def test_program_data_is_preserved_and_secrets_are_not_payload() -> None:
    assert "RemoveFolder" not in WXS
    assert '<Directory Id="PROGRAMDATAROOT" Name="CryptoHunter" />' in WXS
    lowered = WXS.lower()
    assert ".key" not in lowered and "private key" in lowered
    assert "%temp%" not in lowered


def test_no_acceptance_service_or_installer_network() -> None:
    assert "windows_test_service" not in WXS
    assert "windows_stage8_postgresql_service" not in WXS
    body = WXS.split("?>", 1)[1].replace("http://wixtoolset.org/schemas/v4/wxs", "")
    assert not any(word in body.lower() for word in ("http://", "https://", "download"))


def test_postgresql_and_wix_are_exactly_pinned() -> None:
    pins = json.loads((ROOT / "deployment/windows_installer/pins.json").read_text())
    assert pins["wix"]["version"] == CONTRACT.wix_version
    assert pins["postgresql"]["version"] == "17.11"
    assert pins["postgresql"]["packaging_revision"] == "4"
    assert pins["postgresql"]["source_url"].endswith(pins["postgresql"]["archive"])
    assert len(pins["postgresql"]["sha256"]) == 64
    int(pins["postgresql"]["sha256"], 16)
    assert CONTRACT.postgresql_port != 5432
    assert CONTRACT.postgresql_host == "127.0.0.1"


def test_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    archive = tmp_path / "postgresql.zip"
    archive.write_bytes(b"not-postgresql")
    with pytest.raises(InstallerBuildError, match="SHA-256 mismatch"):
        verify_postgresql(archive, "0" * 64)


def _archive(path: Path, names: tuple[str, ...]) -> None:
    with zipfile.ZipFile(path, "w") as bundle:
        for name in names:
            bundle.writestr(name, b"payload")


def test_postgresql_archive_is_normalized_only_from_exact_vendor_root(tmp_path: Path) -> None:
    archive = tmp_path / "pg.zip"
    names = tuple(
        f"pgsql/bin/{name}"
        for name in ("postgres.exe", "pg_ctl.exe", "initdb.exe", "psql.exe", "pg_isready.exe")
    ) + ("pgsql/lib/a.dll", "pgsql/share/a.txt")
    _archive(archive, names)
    destination = tmp_path / "Payload" / "PostgreSQL"
    normalize_postgresql_archive(archive, destination)
    assert (destination / "bin" / "postgres.exe").is_file()
    wrong = tmp_path / "wrong.zip"
    _archive(wrong, tuple(name.replace("pgsql/", "postgres/") for name in names))
    with pytest.raises(InstallerBuildError, match="unexpected EDB"):
        normalize_postgresql_archive(wrong, tmp_path / "bad")


def test_postgresql_archive_rejects_missing_runtime_file(tmp_path: Path) -> None:
    archive = tmp_path / "pg.zip"
    _archive(
        archive,
        (
            "pgsql/bin/postgres.exe",
            "pgsql/bin/initdb.exe",
            "pgsql/bin/psql.exe",
            "pgsql/bin/pg_isready.exe",
            "pgsql/lib/a",
            "pgsql/share/a",
        ),
    )
    with pytest.raises(InstallerBuildError, match="pg_ctl.exe"):
        normalize_postgresql_archive(archive, tmp_path / "bad")


def test_wix_version_is_exact_stable_or_reviewed_metadata() -> None:
    assert canonical_wix_version("7.0.0") == "7.0.0"
    assert canonical_wix_version("7.0.0+build.1") == "7.0.0+build.1"
    for bad in ("7.0.0-preview.1", "7.0.01", "7.0.0 unexpected", "7.0.1"):
        with pytest.raises(InstallerBuildError):
            canonical_wix_version(bad)


def test_canonical_workflow_uses_only_per_invocation_wix_eula_acceptance() -> None:
    workflow = (ROOT / ".github/workflows/platform-deployment.yml").read_text(encoding="utf-8")
    builder = (ROOT / "deployment/windows_installer/build.py").read_text(encoding="utf-8")
    assert "BLOCKED_ON_WIX7_EULA_ACCEPTANCE" not in workflow
    assert re.search(r'"build",\s*"-acceptEula",\s*"wix7",', builder)
    combined = workflow + builder
    assert "wix eula accept wix7" not in combined
    assert "eula accept" not in combined.lower()


def test_failed_msi_install_reports_failure_and_leaves_probe_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    proof_invoked = False

    def fail_msiexec(arguments: list[str], log: Path, timeout: int = 600) -> int:
        raise clean_install.CleanInstallError("msiexec failed with 1603")

    def record_proof(manifest: Path) -> dict[str, str]:
        nonlocal proof_invoked
        proof_invoked = True
        return {}

    monkeypatch.setattr(clean_install, "_msiexec", fail_msiexec)
    monkeypatch.setattr(clean_install, "_proof_install", record_proof)

    with pytest.raises(clean_install.CleanInstallError, match="1603"):
        clean_install._install_and_prove(
            tmp_path / "product.msi",
            tmp_path / "installer-manifest.json",
            tmp_path / "logs" / "install.log",
        )

    output = capsys.readouterr().out
    assert "MSI_INSTALL = FAIL" in output
    assert "MSI_INSTALL = PASS" not in output
    assert not proof_invoked
    assert "CLEAN_INSTALL_PROBE = FAIL" not in output
    assert "CLEAN_INSTALL_PROBE = PASS" not in output


def test_msi_timeout_reports_failure_and_propagates_original_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    timeout = subprocess.TimeoutExpired("msiexec.exe", 600)
    monkeypatch.setattr(
        clean_install, "_msiexec", lambda *args, **kwargs: (_ for _ in ()).throw(timeout)
    )

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        clean_install._install_and_prove(
            tmp_path / "product.msi",
            tmp_path / "installer-manifest.json",
            tmp_path / "logs" / "install.log",
        )

    assert raised.value is timeout
    output = capsys.readouterr().out
    assert "MSI_INSTALL = FAIL" in output
    assert "MSI_INSTALL = PASS" not in output


def test_failed_install_proof_reports_failure_after_successful_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(clean_install, "_msiexec", lambda *args, **kwargs: 0)

    def fail_proof(manifest: Path) -> dict[str, str]:
        raise clean_install.CleanInstallError("proof failed")

    monkeypatch.setattr(clean_install, "_proof_install", fail_proof)

    with pytest.raises(clean_install.CleanInstallError, match="proof failed"):
        clean_install._install_and_prove(
            tmp_path / "product.msi",
            tmp_path / "installer-manifest.json",
            tmp_path / "logs" / "install.log",
        )

    output = capsys.readouterr().out
    assert "MSI_INSTALL = PASS" in output
    assert "CLEAN_INSTALL_PROBE = FAIL" in output
    assert "CLEAN_INSTALL_PROBE = PASS" not in output


def test_unexpected_install_proof_failure_is_reported_and_propagated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    failure = FileNotFoundError("installed manifest disappeared")
    monkeypatch.setattr(clean_install, "_msiexec", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        clean_install, "_proof_install", lambda manifest: (_ for _ in ()).throw(failure)
    )

    with pytest.raises(FileNotFoundError) as raised:
        clean_install._install_and_prove(
            tmp_path / "product.msi",
            tmp_path / "installer-manifest.json",
            tmp_path / "logs" / "install.log",
        )

    assert raised.value is failure
    output = capsys.readouterr().out
    assert "MSI_INSTALL = PASS" in output
    assert "CLEAN_INSTALL_PROBE = FAIL" in output
    assert "CLEAN_INSTALL_PROBE = PASS" not in output


def test_stage9_workflow_always_preserves_msi_diagnostics_without_weakening_gate() -> None:
    workflow = (ROOT / ".github/workflows/platform-deployment.yml").read_text(encoding="utf-8")
    clean_install_job = workflow.split("  windows-clean-install-integration:", 1)[1].split(
        "\n  linux-deployment-integration:", 1
    )[0]
    diagnostic_step = clean_install_job.split("- name: Preserve Stage-9 MSI diagnostics", 1)[1]
    diagnostic_step = diagnostic_step.split("\n      - uses:", 1)[0]
    canonical_step = clean_install_job.split("- name: Build and run canonical Stage-9 proof", 1)[1]
    canonical_step = canonical_step.split("\n      - name:", 1)[0]

    assert "if: always()" in diagnostic_step
    assert "name: windows-clean-install-diagnostics" in diagnostic_step
    assert "dist/windows/logs/*.log" in diagnostic_step
    assert "dist/windows/installer-manifest.json" in diagnostic_step
    assert "if-no-files-found: ignore" in diagnostic_step
    assert "continue-on-error: true" not in canonical_step


@pytest.mark.parametrize(("source", "expected"), [("1.2.3", "1.2.3"), ("1.2.3-rc.1", "1.2.3")])
def test_semver_has_deterministic_legal_msi_mapping(source: str, expected: str) -> None:
    assert msi_version(source) == expected


def test_stage10_is_not_part_of_stage9() -> None:
    combined = WXS + (ROOT / "docs/windows_stage9_installer.md").read_text()
    assert "auto updater" not in combined.lower()
    assert "Stage 10" in combined


def test_builder_owns_all_production_executable_targets() -> None:
    source = (ROOT / "deployment/windows_installer/build.py").read_text()
    for target in (
        "backend_service.py",
        "verifier_service.py",
        "postgresql_service.py",
        "provision.py",
    ):
        assert target in source
        assert (ROOT / "deployment/windows_installer" / target).is_file()
    assert "args.payload" not in source
    assert "windows_test_service" not in source
    assert "windows_stage8_postgresql_service" not in source


def test_transaction_journal_rejects_preexisting_machine_state(tmp_path: Path) -> None:
    program_files = tmp_path / "Program Files" / "CryptoHunter"
    program_files.mkdir(parents=True)
    program_data = tmp_path / "ProgramData" / "CryptoHunter"
    program_data.parent.mkdir(parents=True)
    path, journal = create_journal(program_files, program_data)
    assert journal["schema_version"] == 1
    assert journal["state"] == "PROVISIONING"
    assert path.parent == program_data.parent
    assert journal["service_names"] == [
        "CryptoHunterPostgreSQL",
        "CryptoHunterBackend",
        "CryptoHunterFreshnessVerifier",
    ]
    assert path.is_file()
    program_data.mkdir(parents=True)
    with pytest.raises(Exception, match="pre-exists"):
        create_journal(program_files, program_data)


def test_failed_transaction_rollback_only_removes_journal_owned_state(tmp_path: Path) -> None:
    program_files = tmp_path / "Program Files" / "CryptoHunter"
    program_files.mkdir(parents=True)
    program_data = tmp_path / "ProgramData" / "CryptoHunter"
    program_data.parent.mkdir(parents=True)
    journal_path, journal = create_journal(program_files, program_data)
    program_data.mkdir(parents=True)
    owned = program_data / "Runtime"
    owned.mkdir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    journal["resources_created"] = [str(program_data.resolve()), str(owned.resolve())]
    journal_path.write_text(json.dumps(journal))
    rollback(program_files, program_data)
    assert not program_data.exists()
    assert foreign.is_dir()


def test_provisioned_startservices_failure_is_still_rollback_owned(tmp_path: Path) -> None:
    program_files = tmp_path / "Program Files" / "CryptoHunter"
    program_files.mkdir(parents=True)
    program_data = tmp_path / "ProgramData" / "CryptoHunter"
    program_data.parent.mkdir(parents=True)
    journal_path, journal = create_journal(program_files, program_data)
    program_data.mkdir(parents=True)
    journal["resources_created"] = [str(program_data.resolve())]
    journal["state"] = "PROVISIONED"
    journal_path.write_text(json.dumps(journal))
    rollback(program_files, program_data)
    assert not program_data.exists()
    assert not journal_path.exists()


def test_commit_moves_authority_to_programdata_and_preserves_it(tmp_path: Path) -> None:
    program_files = tmp_path / "Program Files" / "CryptoHunter"
    program_files.mkdir(parents=True)
    program_data = tmp_path / "ProgramData" / "CryptoHunter"
    program_data.parent.mkdir(parents=True)
    journal_path, journal = create_journal(program_files, program_data)
    program_data.mkdir(parents=True)
    journal["state"] = "PROVISIONED"
    journal_path.write_text(json.dumps(journal))
    commit(program_files, program_data)
    assert not journal_path.exists()
    ownership = json.loads((program_data / ".stage9-install-ownership.json").read_text())
    assert ownership["state"] == "COMMITTED"
    rollback(program_files, program_data)
    assert program_data.is_dir()


def test_postgresql_is_created_suspended_and_assigned_before_resume(tmp_path: Path) -> None:
    events = []

    class Handle:
        def Close(self):
            events.append("close")

    process, thread, job = Handle(), Handle(), Handle()

    class ProcessApi:
        STARTUPINFO = Handle

        @staticmethod
        def CreateProcess(*args):
            assert args[5] == 4
            events.append("create-suspended")
            return process, thread, 41, 42

        @staticmethod
        def ResumeThread(value):
            assert value is thread
            events.append("resume")

    class JobApi:
        JobObjectExtendedLimitInformation = 1
        JobObjectBasicProcessIdList = 2
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

        @staticmethod
        def CreateJobObject(*args):
            events.append("job")
            return job

        @staticmethod
        def QueryInformationJobObject(_job, kind):
            return (
                {"BasicLimitInformation": {"LimitFlags": 0}}
                if kind == 1
                else {"ProcessIdList": [41]}
            )

        @staticmethod
        def SetInformationJobObject(*args):
            events.append("kill-on-close")

        @staticmethod
        def AssignProcessToJobObject(*args):
            events.append("assign")

    class Con:
        CREATE_SUSPENDED = 4

    create_suspended_in_job(
        ["postgres.exe"],
        tmp_path,
        win32api=object(),
        win32con=Con,
        win32job=JobApi,
        win32process=ProcessApi,
    )
    assert events.index("assign") < events.index("resume")


def test_postgresql_never_ready_fails_closed_without_sleep_guess(tmp_path: Path) -> None:
    class Result:
        returncode = 1

    class Event:
        WAIT_OBJECT_0 = 0
        WAIT_TIMEOUT = 258

        @staticmethod
        def WaitForSingleObject(handle, timeout):
            return 258

    class ProcessApi:
        @staticmethod
        def GetExitCodeProcess(handle):
            return 259

    times = iter((0.0, 0.0, 2.0))
    with pytest.raises(RuntimeError, match="readiness deadline"):
        wait_ready(
            tmp_path / "pg_isready.exe",
            object(),
            __import__("threading").Event(),
            timeout=1,
            clock=lambda: next(times),
            run=lambda *a, **k: Result(),
            win32event=Event,
            win32process=ProcessApi,
        )


def test_postgresql_exit_before_readiness_reports_real_exit_code(tmp_path: Path) -> None:
    class Event:
        WAIT_OBJECT_0 = 0
        WAIT_TIMEOUT = 258
        WaitForSingleObject = staticmethod(lambda handle, timeout: 0)

    class ProcessApi:
        GetExitCodeProcess = staticmethod(lambda handle: 17)

    with pytest.raises(RuntimeError, match="exit_code=17"):
        wait_ready(
            tmp_path / "pg_isready.exe",
            object(),
            __import__("threading").Event(),
            win32event=Event,
            win32process=ProcessApi,
        )


def test_postgresql_readiness_success_requires_live_process(tmp_path: Path) -> None:
    class Event:
        WAIT_OBJECT_0 = 0
        WAIT_TIMEOUT = 258
        WaitForSingleObject = staticmethod(lambda handle, timeout: 258)

    class ProcessApi:
        GetExitCodeProcess = staticmethod(lambda handle: 259)

    class Result:
        returncode = 0

    wait_ready(
        tmp_path / "pg_isready.exe",
        object(),
        __import__("threading").Event(),
        run=lambda *a, **k: Result(),
        win32event=Event,
        win32process=ProcessApi,
    )


def test_postgresql_clean_stop_requires_signaled_zero_exit() -> None:
    class Event:
        WAIT_OBJECT_0 = 0
        WaitForSingleObject = staticmethod(lambda handle, timeout: 0)

    class ProcessApi:
        GetExitCodeProcess = staticmethod(lambda handle: 0)

    require_clean_exit(object(), timeout_ms=10_000, win32event=Event, win32process=ProcessApi)


def test_probe_has_separate_executable_authority_for_each_install_proof() -> None:
    source = (ROOT / "deployment/windows_stage9_clean_install.py").read_text()
    for name in ("files", "services", "dacl", "postgresql", "mtls_matrix", "backend", "logging"):
        assert f"def prove_{name}" in source
        assert f'proofs["{name}"] = "PASS"' in source
    assert 'return {name: "PASS"' not in source


def test_production_dacl_reuses_frozen_stage4_expected_aces() -> None:
    source = (ROOT / "deployment/windows_installer/provision.py").read_text()
    assert "expected_aces(role, sids[CONTRACT.backend_service])" in source
    assert "FILE_GENERIC_READ | FILE_GENERIC_EXECUTE" in source
    for role in ('"CONFIG"', '"STATE"', '"RUNTIME"', '"LOGS"'):
        assert role in source


def test_transaction_authority_never_uses_program_files() -> None:
    source = (ROOT / "deployment/windows_installer/provision.py").read_text()
    assert "return program_data.parent / TRANSACTION_JOURNAL" in source
    assert "program_data / OWNERSHIP_RECORD" in source
    assert "program_files / JOURNAL" not in source
    assert 'Execute="commit"' in WXS
