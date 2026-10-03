from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from deployment import windows_stage9_clean_install as clean_install
from deployment.windows_installer import provision
from deployment.windows_installer.build import (
    InstallerBuildError,
    canonical_wix_version,
    msi_version,
    normalize_postgresql_archive,
    verify_postgresql,
)
from deployment.windows_installer.contract import CONTRACT
from deployment.windows_installer.provision import (
    _first_exception_type,
    commit,
    create_journal,
    rollback,
)
from deployment.windows_installer.postgresql_service import (
    STARTUP_STAGES,
    create_suspended_in_job,
    main as postgresql_service_main,
    require_clean_exit,
    startup_diagnostic_path,
    try_write_startup_diagnostic,
    wait_ready,
)
from deployment.windows_installer.corehost_composition import (
    WindowsProvisioningAdapterUnavailable,
)
from deployment.windows_stage9_production_trust import ProductionTrustUnavailable


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
        'Name="CryptoHunterBackend"\n             DisplayName="CryptoHunter Backend" Type="ownProcess" Start="demand"'
        in WXS
    )
    backend_control = re.search(r'<ServiceControl Id="BackendControl"[^>]+>', WXS)
    assert backend_control is not None and 'Start="install"' not in backend_control.group()
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
    assert "dist/windows/logs/*.json" in diagnostic_step
    assert "dist/windows/installer-manifest.json" in diagnostic_step
    assert "dist/windows/windows-clean-install-receipt.json" in diagnostic_step
    assert "dist/windows/windows-clean-install-evidence.json" in diagnostic_step
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
        def CreateJobObject(attributes, name):
            assert attributes is None
            if not isinstance(name, str):
                raise TypeError("name must be str")
            events.append("job")
            return job

        @staticmethod
        def QueryInformationJobObject(_job, kind):
            return {"BasicLimitInformation": {"LimitFlags": 0}} if kind == 1 else (41,)

        @staticmethod
        def SetInformationJobObject(*args):
            events.append("kill-on-close")

        @staticmethod
        def AssignProcessToJobObject(*args):
            events.append("assign")

    class Con:
        CREATE_SUSPENDED = 4

    with pytest.raises(TypeError, match="name must be str"):
        JobApi.CreateJobObject(None, None)

    created_job, created_process, created_thread, pid = create_suspended_in_job(
        ["postgres.exe"],
        tmp_path,
        win32api=object(),
        win32con=Con,
        win32job=JobApi,
        win32process=ProcessApi,
        stage=lambda value: events.append(value),
    )
    assert (created_job, created_process, created_thread, pid) == (job, process, thread, 41)
    assert events.index("assign") < events.index("resume")
    boundaries = [value for value in events if value in STARTUP_STAGES]
    assert boundaries == [
        "CREATE_JOB_OBJECT",
        "CONFIGURE_JOB_OBJECT",
        "CREATE_SUSPENDED_PROCESS",
        "ASSIGN_JOB",
        "VERIFY_JOB_MEMBERSHIP",
        "RESUME_PROCESS",
    ]


@pytest.mark.parametrize("observed", [(), (99,), (41, 99), ("malformed",)])
def test_postgresql_job_membership_query_back_fails_closed(
    tmp_path: Path, observed: tuple[object, ...]
) -> None:
    events: list[str] = []

    class Handle:
        def __init__(self) -> None:
            self.closed = False

        def Close(self) -> None:
            self.closed = True

    process, thread, job = Handle(), Handle(), Handle()

    class ProcessApi:
        STARTUPINFO = Handle

        @staticmethod
        def CreateProcess(*_args):
            return process, thread, 41, 42

        @staticmethod
        def ResumeThread(_thread):
            events.append("resume")

    class JobApi:
        JobObjectExtendedLimitInformation = 1
        JobObjectBasicProcessIdList = 2
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

        @staticmethod
        def CreateJobObject(_attributes, _name):
            return job

        @staticmethod
        def QueryInformationJobObject(_job, kind):
            if kind == JobApi.JobObjectExtendedLimitInformation:
                return {"BasicLimitInformation": {"LimitFlags": 0}}
            return observed

        @staticmethod
        def SetInformationJobObject(*_args):
            pass

        @staticmethod
        def AssignProcessToJobObject(*_args):
            pass

    with pytest.raises(RuntimeError):
        create_suspended_in_job(
            ["postgres.exe"],
            tmp_path,
            win32api=object(),
            win32con=SimpleNamespace(CREATE_SUSPENDED=4),
            win32job=JobApi,
            win32process=ProcessApi,
            stage=events.append,
        )
    assert "VERIFY_JOB_MEMBERSHIP" in events
    assert "RESUME_PROCESS" not in events
    assert "resume" not in events
    assert job.closed and process.closed and thread.closed


def test_postgresql_assign_failure_precedes_membership_verification(tmp_path: Path) -> None:
    events: list[str] = []

    class Handle:
        def __init__(self) -> None:
            self.closed = False

        def Close(self) -> None:
            self.closed = True

    process, thread, job = Handle(), Handle(), Handle()

    class ProcessApi:
        STARTUPINFO = Handle

        @staticmethod
        def CreateProcess(*_args):
            return process, thread, 41, 42

        @staticmethod
        def ResumeThread(_thread):
            events.append("resume")

    class JobApi:
        JobObjectExtendedLimitInformation = 1
        JobObjectBasicProcessIdList = 2
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        CreateJobObject = staticmethod(lambda *_args: job)
        QueryInformationJobObject = staticmethod(
            lambda _job, _kind: {"BasicLimitInformation": {"LimitFlags": 0}}
        )
        SetInformationJobObject = staticmethod(lambda *_args: None)

        @staticmethod
        def AssignProcessToJobObject(*_args):
            raise OSError("assign")

    with pytest.raises(OSError, match="assign"):
        create_suspended_in_job(
            ["postgres.exe"],
            tmp_path,
            win32api=object(),
            win32con=SimpleNamespace(CREATE_SUSPENDED=4),
            win32job=JobApi,
            win32process=ProcessApi,
            stage=events.append,
        )
    assert "ASSIGN_JOB" in events
    assert "VERIFY_JOB_MEMBERSHIP" not in events
    assert "RESUME_PROCESS" not in events
    assert "resume" not in events
    assert job.closed and process.closed and thread.closed


@pytest.mark.parametrize("failure_point", ["query", "set", "startup"])
def test_postgresql_job_is_closed_when_setup_fails(tmp_path: Path, failure_point: str) -> None:
    class Handle:
        closed = False

        def Close(self):
            self.closed = True

    job = Handle()

    class JobApi:
        JobObjectExtendedLimitInformation = 1
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

        @staticmethod
        def CreateJobObject(attributes, name):
            assert (attributes, name) == (None, "")
            return job

        @staticmethod
        def QueryInformationJobObject(*_args):
            if failure_point == "query":
                raise OSError("query")
            return {"BasicLimitInformation": {"LimitFlags": 0}}

        @staticmethod
        def SetInformationJobObject(*_args):
            if failure_point == "set":
                raise OSError("set")

    class ProcessApi:
        @staticmethod
        def STARTUPINFO():
            raise OSError("startup")

    with pytest.raises(OSError, match=failure_point):
        create_suspended_in_job(
            ["postgres.exe"],
            tmp_path,
            win32api=object(),
            win32con=object(),
            win32job=JobApi,
            win32process=ProcessApi,
        )
    assert job.closed


def test_service_startup_diagnostic_is_allowlisted_and_secret_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ProgramData", str(tmp_path))
    secret = "password=private-key-material"
    startup_diagnostic_path().write_text("install window", encoding="utf-8")
    assert try_write_startup_diagnostic("WAIT_READY", RuntimeError(secret), 23)
    value = json.loads((tmp_path / ".CryptoHunter.postgresql-service-startup.json").read_text())
    assert value == {
        "service": "CryptoHunterPostgreSQL",
        "stage": "WAIT_READY",
        "first_exception": "RuntimeError",
        "child_exit_code": 23,
    }
    assert secret not in json.dumps(value)


def test_post_commit_service_restart_does_not_recreate_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    program_files = tmp_path / "Program Files" / "CryptoHunter"
    program_files.mkdir(parents=True)
    program_data = tmp_path / "ProgramData" / "CryptoHunter"
    program_data.parent.mkdir(parents=True)
    monkeypatch.setenv("ProgramData", str(program_data.parent))
    journal_path, journal = create_journal(program_files, program_data)
    program_data.mkdir()
    journal["state"] = "PROVISIONED"
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    startup_diagnostic_path().write_text("install window", encoding="utf-8")
    assert startup_diagnostic_path().is_file()
    commit(program_files, program_data)
    assert not startup_diagnostic_path().exists()
    calls = []
    monkeypatch.setattr(
        "deployment.windows_installer.postgresql_service.run_service",
        lambda name, worker: calls.append((name, worker)),
    )

    postgresql_service_main()

    assert calls and calls[0][0] == "CryptoHunterPostgreSQL"
    assert not startup_diagnostic_path().exists()


def test_telemetry_write_failure_does_not_prevent_dispatcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ProgramData", str(tmp_path))
    diagnostic = startup_diagnostic_path()
    diagnostic.write_text("install window", encoding="utf-8")
    monkeypatch.setattr(
        Path, "open", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("denied"))
    )
    dispatched = []
    monkeypatch.setattr(
        "deployment.windows_installer.postgresql_service.run_service",
        lambda name, worker: dispatched.append(name),
    )

    postgresql_service_main()

    assert dispatched == ["CryptoHunterPostgreSQL"]


def test_dispatcher_failure_records_exact_exception_without_masking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ProgramData", str(tmp_path))
    startup_diagnostic_path().write_text("install window", encoding="utf-8")
    original = OSError("dispatcher detail must not be persisted")
    monkeypatch.setattr(
        "deployment.windows_installer.postgresql_service.run_service",
        lambda *_args: (_ for _ in ()).throw(original),
    )

    with pytest.raises(OSError) as raised:
        postgresql_service_main()

    assert raised.value is original
    assert json.loads(startup_diagnostic_path().read_text(encoding="utf-8")) == {
        "service": "CryptoHunterPostgreSQL",
        "stage": "SCM_DISPATCHER",
        "first_exception": "OSError",
    }


def test_dispatcher_telemetry_failure_does_not_mask_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = OSError("dispatcher")
    monkeypatch.setattr(
        "deployment.windows_installer.postgresql_service.try_write_startup_diagnostic",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        "deployment.windows_installer.postgresql_service.run_service",
        lambda *_args: (_ for _ in ()).throw(original),
    )
    with pytest.raises(OSError) as raised:
        postgresql_service_main()
    assert raised.value is original


@pytest.mark.parametrize(
    ("identity", "accepted"),
    [
        ("CryptoHunterPostgreSQL", True),
        ("CryptoHunter Private PostgreSQL", True),
        ("UnrelatedService", False),
    ],
)
def test_scm_event_collection_requires_exact_service_identity(
    identity: str, accepted: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = subprocess.CompletedProcess(
        [], 0, json.dumps([{"id": 7000, "properties": [identity, "%%193"]}]), ""
    )
    monkeypatch.setattr(clean_install.subprocess, "run", lambda *args, **kwargs: result)
    assert bool(clean_install._collect_scm_events()) is accepted


@pytest.mark.parametrize(
    ("diagnostic", "events", "expected"),
    [
        (
            {"stage": "CREATE_JOB_OBJECT", "first_exception": "TypeError"},
            [],
            "JOB_OBJECT_SETUP_FAILURE",
        ),
        (
            {"stage": "CONFIGURE_JOB_OBJECT", "first_exception": "OSError"},
            [],
            "JOB_OBJECT_SETUP_FAILURE",
        ),
        (
            {"stage": "CREATE_SUSPENDED_PROCESS", "first_exception": "OSError"},
            [],
            "POSTGRES_CREATEPROCESS_FAILURE",
        ),
        (
            {"stage": "SCM_DISPATCHER", "first_exception": "OSError"},
            [],
            "SERVICE_HOST_START_FAILURE",
        ),
        ({"stage": "ASSIGN_JOB", "first_exception": "OSError"}, [], "JOB_ASSIGNMENT_FAILURE"),
        (
            {"stage": "VERIFY_JOB_MEMBERSHIP", "first_exception": "ValueError"},
            [],
            "JOB_MEMBERSHIP_VERIFICATION_FAILURE",
        ),
        (
            {"stage": "WAIT_READY", "first_exception": "RuntimeError", "child_exit_code": 1},
            [],
            "POSTGRES_EARLY_EXIT",
        ),
        (
            {"stage": "WAIT_READY", "first_exception": "PostgreSQLReadinessTimeout"},
            [],
            "POSTGRES_READINESS_TIMEOUT",
        ),
        ({"stage": "WAIT_READY", "first_exception": "OSError"}, [], "UNCLASSIFIED"),
        ({"stage": "REPORT_RUNNING", "first_exception": "OSError"}, [], "SCM_REPORTING_FAILURE"),
        (None, [{"id": 7038, "properties": ["CryptoHunterPostgreSQL"]}], "SERVICE_LOGON_FAILURE"),
        (
            None,
            [{"id": 7000, "properties": ["CryptoHunterPostgreSQL", "%%193"]}],
            "SERVICE_HOST_START_FAILURE",
        ),
        (None, [{"id": 7000, "properties": ["CryptoHunterPostgreSQL", "%%1067"]}], "UNCLASSIFIED"),
    ],
)
def test_native_start_failure_categories_require_defining_evidence(
    diagnostic: dict[str, object] | None,
    events: list[dict[str, object]],
    expected: str,
) -> None:
    if diagnostic is not None:
        diagnostic = {"service": "CryptoHunterPostgreSQL", **diagnostic}
    assert clean_install.classify_service_start_failure(diagnostic, events) == expected


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
    for name in (
        "files",
        "services",
        "dacl",
        "postgresql",
        "authority_absent",
        "production_enrollment_fail_closed",
    ):
        assert f"def prove_{name}" in source
        assert f'proofs["{name}"] = "PASS"' in source
    assert 'return {name: "PASS"' not in source


def test_msi_install_never_invokes_production_enrollment() -> None:
    provision = (ROOT / "deployment/windows_installer/provision.py").read_text()
    install_body = provision.split("def install(", 1)[1].split("def enroll(", 1)[0]
    assert "load_windows_external_provisioning_handoff" not in install_body
    assert "materialize_canonical_pre_state" not in install_body
    assert "enroll" not in WXS.lower()


def test_enrollment_is_explicit_fail_closed_and_only_then_activates_backend() -> None:
    provision = (ROOT / "deployment/windows_installer/provision.py").read_text()
    enroll_body = provision.split("def enroll(", 1)[1].split("def rollback(", 1)[0]
    assert enroll_body.index("load_windows_external_provisioning_handoff()") < enroll_body.index(
        "materialize_canonical_pre_state"
    )
    assert enroll_body.index("materialize_canonical_pre_state") < enroll_body.index(
        "_commit_backend_start_type()"
    )
    assert "TEST_ONLY" not in enroll_body and "development" not in enroll_body.lower()


def test_clean_install_never_claims_production_enrollment_passed() -> None:
    source = (ROOT / "deployment/windows_stage9_clean_install.py").read_text()
    assert 'proofs["production_enrollment"] = "PASS"' not in source
    assert 'proofs["production_enrollment_fail_closed"] = "PASS"' in source
    assert '"post_enrollment_live_qualification": POST_ENROLLMENT_QUALIFICATION_STATE' in source
    assert "POST_ENROLLMENT_LIVE_QUALIFICATION = REQUIRED" in source


def test_safe_diagnostic_identifies_first_exception_without_rendering_it() -> None:
    secret = RuntimeError("private-ceremony-material")
    try:
        try:
            raise secret
        except RuntimeError:
            raise OSError("cleanup failure")
    except OSError as outer:
        assert _first_exception_type(outer) == "RuntimeError"


def test_enroll_cli_reports_missing_production_trust_as_root_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_before_legal_enrollment(program_files: Path, program_data: Path) -> None:
        del program_files, program_data
        try:
            raise ProductionTrustUnavailable("PRODUCTION_TRUST_UNAVAILABLE")
        except ProductionTrustUnavailable as exc:
            raise WindowsProvisioningAdapterUnavailable("PRODUCTION_TRUST_UNAVAILABLE") from exc

    monkeypatch.setattr(provision, "enroll", fail_before_legal_enrollment)
    monkeypatch.setattr(provision, "os", SimpleNamespace(name="nt"))

    assert (
        provision.main(
            [
                "enroll",
                "--program-files",
                str(tmp_path / "ProgramFiles"),
                "--program-data",
                str(tmp_path / "ProgramData"),
            ]
        )
        == 1
    )
    assert capsys.readouterr().err.splitlines() == [
        "provisioning failed: operation=enroll first_exception=ProductionTrustUnavailable"
    ]


def test_safe_failure_is_secret_free_atomic_and_cannot_be_redirected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    machine_root = tmp_path / "ProgramData"
    machine_root.mkdir()
    monkeypatch.setenv("ProgramData", str(machine_root))
    secret = "password=correct-horse-private-key"

    provision.write_safe_failure("install", "POSTGRES_INITDB", RuntimeError(secret))

    path = machine_root / provision.SAFE_FAILURE_DIAGNOSTIC
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value == {
        "schema_version": 1,
        "operation": "install",
        "stage": "POSTGRES_INITDB",
        "first_exception": "RuntimeError",
    }
    assert secret not in path.read_text(encoding="utf-8")
    assert not path.with_suffix(".tmp").exists()
    with pytest.raises(provision.ProvisionError, match="not canonical"):
        provision.read_safe_failure(tmp_path / "caller-selected.json")


def test_install_failure_diagnostic_survives_rollback_and_commit_clears_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    machine_root = tmp_path / "ProgramData"
    machine_root.mkdir()
    monkeypatch.setenv("ProgramData", str(machine_root))
    monkeypatch.setattr(
        provision,
        "os",
        SimpleNamespace(name="nt", environ=os.environ, replace=os.replace),
    )
    monkeypatch.setattr(
        provision,
        "install",
        lambda pf, pd, stage: (
            stage("POSTGRES_BOOTSTRAP_START"),
            (_ for _ in ()).throw(subprocess.CalledProcessError(1, "secret-command")),
        ),
    )
    common = ["--program-files", str(tmp_path / "pf"), "--program-data", str(tmp_path / "pd")]
    assert provision.main(["install", *common]) == 1
    diagnostic = provision.safe_failure_path()
    assert diagnostic.is_file()

    monkeypatch.setattr(provision, "rollback", lambda *args: None)
    assert provision.main(["rollback", *common]) == 0
    assert diagnostic.is_file()

    monkeypatch.setattr(provision, "commit", lambda *args: None)
    assert provision.main(["commit", *common]) == 0
    assert not diagnostic.exists()


@pytest.mark.parametrize(
    "expected",
    (
        "POSTGRES_INITDB",
        "POSTGRES_BOOTSTRAP_START",
        "POSTGRES_CREATE_DATABASE",
        "POSTGRES_PROVISION_FRESHNESS_AUTHORITY",
        "POSTGRES_QUALIFY_FRESHNESS_AUTHORITY",
        "POSTGRES_FINAL_HBA_WRITE",
        "POSTGRES_FINAL_HBA_QUALIFY",
        "POSTGRES_FINAL_ADMIN_REJECTION",
        "POSTGRES_BOOTSTRAP_STOP",
    ),
)
def test_database_effect_boundaries_set_exact_stage_before_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expected: str
) -> None:
    data = tmp_path / "data"
    security = tmp_path / "security"
    data.mkdir()
    security.mkdir()
    (data / "postgresql.conf").write_text("", encoding="utf-8")
    (security / "ca.crt").write_text("ca", encoding="utf-8")
    (security / "server").mkdir()
    (security / "server" / "server.crt").write_text("cert", encoding="utf-8")
    (security / "server" / "server.key").write_text("key", encoding="utf-8")

    class Connection:
        statement = ""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def execute(self, statement: str):
            self.statement = statement
            return self

        def fetchall(self):
            return []

        def fetchone(self):
            if self.statement == "SELECT 1":
                return (1,)
            return ("postgres", "postgres")

    class OperationalError(Exception):
        pass

    connections = 0

    def connect(*args, **kwargs):
        nonlocal connections
        del args, kwargs
        connections += 1
        if connections == 3:
            raise OperationalError("connection requires a valid client certificate")
        return Connection()

    monkeypatch.setattr(provision.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setitem(
        sys.modules,
        "psycopg",
        SimpleNamespace(connect=connect, OperationalError=OperationalError),
    )
    monkeypatch.setattr(provision, "provision_postgresql_freshness_authority", lambda *args: None)
    monkeypatch.setattr(provision, "qualify_postgresql_freshness_authority", lambda *args: None)
    monkeypatch.setattr(provision, "qualify_hba_rows", lambda *args: None)

    def inject(stage: str) -> None:
        if stage == expected:
            raise RuntimeError("controlled failure with secret text")

    with pytest.raises(RuntimeError, match="controlled failure"):
        provision._configure_database(tmp_path, data, security, inject)


def test_final_hba_cutover_reuses_admin_session_and_rejects_new_admin_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    security = tmp_path / "security"
    data.mkdir()
    (security / "server").mkdir(parents=True)
    (data / "postgresql.conf").write_text("", encoding="utf-8")
    for path in (
        security / "ca.crt",
        security / "server" / "server.crt",
        security / "server" / "server.key",
    ):
        path.write_text("x", encoding="utf-8")

    events: list[str] = []
    cutover = False

    class OperationalError(Exception):
        pass

    class Result:
        def __init__(self, row=None, rows=None):
            self.row = row
            self.rows = rows

        def fetchone(self):
            return self.row

        def fetchall(self):
            return self.rows

    class Connection:
        def __init__(self, name: str):
            self.name = name

        def __enter__(self):
            events.append(f"enter:{self.name}")
            return self

        def __exit__(self, *_args):
            events.append(f"close:{self.name}")

        def execute(self, statement: str):
            nonlocal cutover
            if "session_user" in statement:
                events.append("admin-session-active")
                return Result(("postgres", "postgres"))
            if "pg_hba_file_rules" in statement:
                assert cutover
                events.append("final-parser-qualification")
                return Result(rows=_valid_hba_rows_for_stage9())
            if statement == "SELECT 1":
                events.append("server-live-on-existing-session")
                return Result((1,))
            return Result()

    connect_count = 0

    def connect(*_args, **_kwargs):
        nonlocal connect_count
        connect_count += 1
        if cutover:
            events.append("new-admin-rejected")
            raise OperationalError("connection requires a valid client certificate")
        return Connection(f"connection-{connect_count}")

    original_write_text = Path.write_text

    def write_text(path: Path, value: str, *args, **kwargs):
        nonlocal cutover
        result = original_write_text(path, value, *args, **kwargs)
        if path == data / "pg_hba.conf" and value != "host all postgres 127.0.0.1/32 trust\n":
            cutover = True
            events.append("final-hba-written")
        return result

    commands: list[str] = []

    def run(command, **_kwargs):
        commands.append(command[1])

    monkeypatch.setattr(Path, "write_text", write_text)
    monkeypatch.setattr(provision.subprocess, "run", run)
    monkeypatch.setitem(
        sys.modules,
        "psycopg",
        SimpleNamespace(connect=connect, OperationalError=OperationalError),
    )
    monkeypatch.setattr(provision, "provision_postgresql_freshness_authority", lambda *_: None)
    monkeypatch.setattr(provision, "qualify_postgresql_freshness_authority", lambda *_: None)

    provision._configure_database(tmp_path, data, security, events.append)

    assert connect_count == 3  # database creation, retained admin, rejected proof
    assert events.index("admin-session-active") < events.index("final-hba-written")
    assert events.index("final-hba-written") < events.index("final-parser-qualification")
    assert events.index("new-admin-rejected") < events.index("server-live-on-existing-session")
    assert events.index("server-live-on-existing-session") < events.index("close:connection-2")
    assert commands[-1] == "stop"
    final_hba = (data / "pg_hba.conf").read_text(encoding="utf-8").splitlines()
    assert final_hba == list(provision.FINAL_HBA)
    assert not any("trust" in line for line in final_hba)


def _valid_hba_rows_for_stage9() -> list[tuple[object, ...]]:
    # Use the shared contract to avoid weakening the exact production expectations.
    database = "freshness_gate"
    return [
        (
            1,
            "hostssl",
            [database],
            ["freshness_crypto_verifier"],
            "127.0.0.1",
            "255.255.255.255",
            "cert",
            ["map=stage8_cert", "clientcert=verify-full"],
            None,
        ),
        (
            2,
            "hostssl",
            [database],
            ["freshness_runtime"],
            "127.0.0.1",
            "255.255.255.255",
            "cert",
            ["map=stage8_cert", "clientcert=verify-full"],
            None,
        ),
        (3, "host", [database], ["all"], "127.0.0.1", "255.255.255.255", "reject", None, None),
        (4, "host", ["all"], ["all"], "127.0.0.1", "255.255.255.255", "reject", None, None),
        (5, "host", ["all"], ["all"], "0.0.0.0", "0.0.0.0", "reject", None, None),
        (6, "host", ["all"], ["all"], "::", "::", "reject", None, None),
    ]


def test_clean_install_copies_and_prints_only_safe_failure_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    machine_root = tmp_path / "ProgramData"
    logs = tmp_path / "dist" / "windows" / "logs"
    machine_root.mkdir()
    logs.mkdir(parents=True)
    monkeypatch.setenv("ProgramData", str(machine_root))
    provision.write_safe_failure("install", "POSTGRES_INITDB", ValueError("dsn=secret"))
    monkeypatch.setattr(
        clean_install, "_msiexec", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError())
    )

    with pytest.raises(RuntimeError):
        clean_install._install_and_prove(
            tmp_path / "x.msi", tmp_path / "manifest.json", logs / "install.log"
        )

    copied = json.loads((logs / "provision-failure.json").read_text(encoding="utf-8"))
    assert copied["stage"] == "POSTGRES_INITDB"
    assert copied["first_exception"] == "ValueError"
    assert "secret" not in json.dumps(copied)
    output = capsys.readouterr().out
    assert "PROVISION_FAILURE_STAGE = POSTGRES_INITDB" in output
    assert "PROVISION_FAILURE_EXCEPTION = ValueError" in output


def test_fail_closed_proof_rejects_unexpected_first_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    machine = tmp_path / "machine"
    (machine / "State").mkdir(parents=True)

    class Result:
        returncode = 1
        stderr = b"provisioning failed: operation=enroll first_exception=FileNotFoundError\n"

    monkeypatch.setattr(clean_install.subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(clean_install.CleanInstallError, match="expected cause"):
        clean_install.prove_production_enrollment_fail_closed(root, machine)


def test_fail_closed_proof_accepts_only_exact_missing_production_trust_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    machine = tmp_path / "machine"
    (machine / "State").mkdir(parents=True)

    class Result:
        returncode = 1
        stderr = (
            b"provisioning failed: operation=enroll first_exception=ProductionTrustUnavailable\n"
        )

    monkeypatch.setattr(clean_install.subprocess, "run", lambda *args, **kwargs: Result())
    clean_install.prove_production_enrollment_fail_closed(root, machine)


def _installed_machine(tmp_path: Path) -> tuple[Path, Path]:
    program_files = tmp_path / "Program Files" / "CryptoHunter"
    program_data = tmp_path / "ProgramData" / "CryptoHunter"
    program_files.mkdir(parents=True)
    (program_data / "State").mkdir(parents=True)
    (program_data / provision.OWNERSHIP_RECORD).write_text(
        json.dumps(
            {
                "state": "COMMITTED",
                "program_data": str(program_data.resolve()),
            }
        )
    )
    return program_files, program_data


@pytest.mark.parametrize(
    "failure_step",
    ("materialize", "start_type", "recovery", "start_service"),
)
def test_enrollment_finalization_resumes_after_each_effect_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_step: str
) -> None:
    program_files, program_data = _installed_machine(tmp_path)
    accepted_claim = __import__(
        "tests.persistence.test_durable_first_run_bootstrap", fromlist=["claim"]
    ).claim()
    provider = object()
    calls: list[str] = []
    failed = False

    monkeypatch.setattr(provision, "load_windows_external_provisioning_handoff", lambda: provider)
    monkeypatch.setattr(
        provision,
        "resolve_accepted_handoff",
        lambda value: (accepted_claim.claim_fingerprint_sha256, accepted_claim, object()),
    )

    def effect(name: str):
        def run(*args):
            nonlocal failed
            calls.append(name)
            if name == failure_step and not failed:
                failed = True
                raise OSError("injected crash")
            if name == "materialize":
                (program_data / "State" / "corehost.sqlite").touch()

        return run

    monkeypatch.setattr(provision, "materialize_canonical_pre_state", effect("materialize"))
    monkeypatch.setattr(provision, "_commit_backend_start_type", effect("start_type"))
    monkeypatch.setattr(provision, "_recovery", effect("recovery"))
    monkeypatch.setattr(provision, "_ensure_backend_running", effect("start_service"))
    monkeypatch.setattr(provision, "_verify_backend_activation", effect("verify"))

    with pytest.raises(OSError, match="injected crash"):
        provision.enroll(program_files, program_data)
    provision.enroll(program_files, program_data)
    record = json.loads((program_data / "State" / provision.ENROLLMENT_FINALIZATION).read_text())
    assert record["phase"] == "COMMITTED"
    assert calls.count(failure_step) == 2


def test_committed_enrollment_is_idempotently_reverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    program_files, program_data = _installed_machine(tmp_path)
    accepted_claim = __import__(
        "tests.persistence.test_durable_first_run_bootstrap", fromlist=["claim"]
    ).claim()
    provider = object()
    monkeypatch.setattr(provision, "load_windows_external_provisioning_handoff", lambda: provider)
    monkeypatch.setattr(
        provision,
        "resolve_accepted_handoff",
        lambda value: (accepted_claim.claim_fingerprint_sha256, accepted_claim, object()),
    )
    for name in (
        "materialize_canonical_pre_state",
        "_commit_backend_start_type",
        "_recovery",
        "_ensure_backend_running",
        "_verify_backend_activation",
    ):
        monkeypatch.setattr(provision, name, lambda *args: None)
    provision.enroll(program_files, program_data)
    provision.enroll(program_files, program_data)
    record = json.loads((program_data / "State" / provision.ENROLLMENT_FINALIZATION).read_text())
    assert record["phase"] == "COMMITTED"


def test_enrollment_finalization_rejects_conflicting_durable_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    program_files, program_data = _installed_machine(tmp_path)
    accepted_claim = __import__(
        "tests.persistence.test_durable_first_run_bootstrap", fromlist=["claim"]
    ).claim()
    path = program_data / "State" / provision.ENROLLMENT_FINALIZATION
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "claim_reference": "f" * 64,
                "account_id": accepted_claim.account_id,
                "device_installation_id": accepted_claim.device_installation_id,
                "phase": "PRE_MATERIALIZED",
            }
        )
    )
    monkeypatch.setattr(provision, "load_windows_external_provisioning_handoff", lambda: object())
    monkeypatch.setattr(
        provision,
        "resolve_accepted_handoff",
        lambda value: (accepted_claim.claim_fingerprint_sha256, accepted_claim, object()),
    )
    with pytest.raises(provision.ProvisionError, match="identity conflict"):
        provision.enroll(program_files, program_data)


@pytest.mark.parametrize("initial_start_type", ("auto", "demand"))
def test_backend_start_type_uses_minimum_query_and_change_rights(
    monkeypatch: pytest.MonkeyPatch, initial_start_type: str
) -> None:
    calls: list[object] = []
    state = {"start_type": 2 if initial_start_type == "auto" else 3}

    def open_service(manager, name, access):
        calls.append(("access", access))
        assert access & 0x0001  # SERVICE_QUERY_CONFIG
        assert access & 0x0002  # SERVICE_CHANGE_CONFIG
        assert not access & 0x0010  # SERVICE_START
        return object()

    def query_config(service):
        calls.append("query")
        return (None, state["start_type"])

    def change_config(service, service_type, start_type, *args):
        calls.append("change")
        state["start_type"] = start_type

    fake = SimpleNamespace(
        SERVICE_QUERY_CONFIG=0x0001,
        SERVICE_CHANGE_CONFIG=0x0002,
        SERVICE_START=0x0010,
        SERVICE_AUTO_START=2,
        SERVICE_NO_CHANGE=-1,
        SC_MANAGER_CONNECT=1,
        OpenSCManager=lambda *args: object(),
        OpenService=open_service,
        QueryServiceConfig=query_config,
        ChangeServiceConfig=change_config,
        CloseServiceHandle=lambda handle: None,
    )
    monkeypatch.setitem(sys.modules, "win32service", fake)
    provision._commit_backend_start_type()
    assert state["start_type"] == fake.SERVICE_AUTO_START
    assert calls.count("query") == 2
    assert calls.count("change") == (0 if initial_start_type == "auto" else 1)


@pytest.mark.parametrize(
    ("checkpoint", "prior_phase"),
    (
        ("PRE_MATERIALIZED", "AUTHORITY_ACCEPTED"),
        ("BACKEND_START_TYPE_COMMITTED", "PRE_MATERIALIZED"),
        ("RECOVERY_POLICY_COMMITTED", "BACKEND_START_TYPE_COMMITTED"),
        ("BACKEND_RUNNING_VERIFIED", "RECOVERY_POLICY_COMMITTED"),
    ),
)
def test_enrollment_retry_reconciles_after_effect_before_checkpoint_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: str,
    prior_phase: str,
) -> None:
    program_files, program_data = _installed_machine(tmp_path)
    accepted_claim = __import__(
        "tests.persistence.test_durable_first_run_bootstrap", fromlist=["claim"]
    ).claim()
    provider = object()
    physical = {
        "pre": False,
        "start_type": "demand",
        "recovery": False,
        "running": False,
    }
    observations = {"pre_reused": 0, "start_changes": 0, "starts": 0}
    monkeypatch.setattr(provision, "load_windows_external_provisioning_handoff", lambda: provider)
    monkeypatch.setattr(
        provision,
        "resolve_accepted_handoff",
        lambda value: (accepted_claim.claim_fingerprint_sha256, accepted_claim, object()),
    )

    def materialize(*args):
        if physical["pre"]:
            observations["pre_reused"] += 1
        physical["pre"] = True

    def start_type():
        if physical["start_type"] != "auto":
            physical["start_type"] = "auto"
            observations["start_changes"] += 1

    def recovery():
        physical["recovery"] = True

    def start_service():
        if not physical["running"]:
            physical["running"] = True
            observations["starts"] += 1

    def verify():
        assert physical == {
            "pre": True,
            "start_type": "auto",
            "recovery": True,
            "running": True,
        }

    monkeypatch.setattr(provision, "materialize_canonical_pre_state", materialize)
    monkeypatch.setattr(provision, "_commit_backend_start_type", start_type)
    monkeypatch.setattr(provision, "_recovery", recovery)
    monkeypatch.setattr(provision, "_ensure_backend_running", start_service)
    monkeypatch.setattr(provision, "_verify_backend_activation", verify)
    real_write_phase = provision._write_enrollment_phase
    crashed = False

    def crash_before_checkpoint(path, record, phase):
        nonlocal crashed
        if phase == checkpoint and not crashed:
            crashed = True
            raise OSError("after-effect crash before checkpoint")
        real_write_phase(path, record, phase)

    monkeypatch.setattr(provision, "_write_enrollment_phase", crash_before_checkpoint)
    with pytest.raises(OSError, match="after-effect"):
        provision.enroll(program_files, program_data)
    durable = json.loads((program_data / "State" / provision.ENROLLMENT_FINALIZATION).read_text())
    assert durable["phase"] == prior_phase

    state_at_crash = {
        "PRE_MATERIALIZED": "pre",
        "BACKEND_START_TYPE_COMMITTED": "start_type",
        "RECOVERY_POLICY_COMMITTED": "recovery",
        "BACKEND_RUNNING_VERIFIED": "running",
    }[checkpoint]
    assert physical[state_at_crash] is True or physical[state_at_crash] == "auto"
    provision.enroll(program_files, program_data)
    durable = json.loads((program_data / "State" / provision.ENROLLMENT_FINALIZATION).read_text())
    assert durable["phase"] == "COMMITTED"
    assert observations["start_changes"] == 1
    assert observations["starts"] == 1
    if checkpoint == "PRE_MATERIALIZED":
        assert observations["pre_reused"] == 1


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
