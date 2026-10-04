from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from deployment.windows_installer.build import installed_payload_digest
from deployment.windows_stage9_pre_enrollment_reset import (
    AUTHORITY_MARKERS,
    BASE_TOP_LEVEL,
    ResetError,
    main,
    qualify_artifact,
    qualify_installed_payload,
    qualify_output_path,
    qualify_preserved_program_data,
    qualify_services,
    query_services,
    require_product_unregistered,
    run,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    msi = tmp_path / "CryptoHunter.msi"
    msi.write_bytes(b"msi")
    files = {
        "CryptoHunterBackend.exe": "1" * 64,
        "CryptoHunterFreshnessVerifier.exe": "2" * 64,
        "CryptoHunterProvision.exe": "3" * 64,
        "PostgreSQL/CryptoHunterPostgreSQL.exe": "4" * 64,
    }
    manifest = tmp_path / "installer-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "architecture": "x64",
                "msi": {"file": msi.name, "sha256": _sha(msi)},
                "installed_files": files,
                "installed_payload_sha256": installed_payload_digest(files),
            }
        ),
        encoding="utf-8",
    )
    return msi, manifest, files


def machine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    program_files = tmp_path / "Program Files" / "CryptoHunter"
    program_data = tmp_path / "ProgramData" / "CryptoHunter"
    program_files.mkdir(parents=True)
    program_data.mkdir(parents=True)
    monkeypatch.setenv("ProgramFiles", str(program_files.parent))
    monkeypatch.setenv("ProgramData", str(program_data.parent))
    for name in BASE_TOP_LEVEL - {".stage9-install-ownership.json"}:
        (program_data / name).mkdir()
    record = {
        "schema_version": 1,
        "state": "COMMITTED",
        "program_files": str(program_files.resolve()),
        "program_data": str(program_data.resolve()),
        "service_names": [
            "CryptoHunterPostgreSQL",
            "CryptoHunterBackend",
            "CryptoHunterFreshnessVerifier",
        ],
        "resources_created": [str(program_data.resolve())],
    }
    (program_data / ".stage9-install-ownership.json").write_text(json.dumps(record))
    return program_files, program_data


def services(program_files: Path):
    return {
        "CryptoHunterPostgreSQL": {
            "PathName": str(program_files / "PostgreSQL/CryptoHunterPostgreSQL.exe"),
            "StartMode": "Auto",
            "StartName": r"NT SERVICE\CryptoHunterPostgreSQL",
            "State": "Running",
            "Dependencies": [],
        },
        "CryptoHunterBackend": {
            "PathName": str(program_files / "CryptoHunterBackend.exe"),
            "StartMode": "Manual",
            "StartName": r"NT SERVICE\CryptoHunterBackend",
            "State": "Stopped",
            "Dependencies": ["CryptoHunterPostgreSQL"],
        },
        "CryptoHunterFreshnessVerifier": {
            "PathName": str(program_files / "CryptoHunterFreshnessVerifier.exe"),
            "StartMode": "Manual",
            "StartName": r"NT SERVICE\CryptoHunterFreshnessVerifier",
            "State": "Stopped",
            "Dependencies": [],
        },
    }


def test_exact_artifact_and_payload_qualify(tmp_path: Path):
    msi, manifest, files = artifact(tmp_path)
    assert qualify_artifact(msi, manifest)[1] == files
    root = tmp_path / "installed"
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        files[name] = _sha(path)
    qualify_installed_payload(root, files)
    (root / "extra").write_bytes(b"no")
    with pytest.raises(ResetError, match="file set"):
        qualify_installed_payload(root, files)


@pytest.mark.parametrize(
    "field,value", [("state", "PROVISIONED"), ("program_data", "C:/wrong"), ("service_names", [])]
)
def test_ownership_contract_rejects_differences(tmp_path, monkeypatch, field, value):
    pf, pd = machine(tmp_path, monkeypatch)
    path = pd / ".stage9-install-ownership.json"
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ResetError, match="ownership"):
        qualify_preserved_program_data(pf, pd)


@pytest.mark.parametrize("marker", AUTHORITY_MARKERS)
def test_every_reviewed_authority_marker_blocks(tmp_path, monkeypatch, marker):
    pf, pd = machine(tmp_path, monkeypatch)
    target = pd / marker
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch() if target.suffix else target.mkdir()
    with pytest.raises(ResetError, match="authority/enrollment"):
        qualify_preserved_program_data(pf, pd)


def test_extra_top_level_and_nested_symlink_block(tmp_path, monkeypatch):
    pf, pd = machine(tmp_path, monkeypatch)
    (pd / "foreign").touch()
    with pytest.raises(ResetError, match="shape"):
        qualify_preserved_program_data(pf, pd)
    (pd / "foreign").unlink()
    (pd / "State" / "link").symlink_to(tmp_path)
    with pytest.raises(ResetError, match="reparse"):
        qualify_preserved_program_data(pf, pd)


@pytest.mark.parametrize(
    "service,field,value",
    [
        ("CryptoHunterBackend", "State", "Running"),
        ("CryptoHunterFreshnessVerifier", "State", "Running"),
        ("CryptoHunterPostgreSQL", "StartName", "LocalSystem"),
        ("CryptoHunterBackend", "Dependencies", []),
    ],
)
def test_service_contract_is_exact(tmp_path, service, field, value):
    pf = tmp_path / "CryptoHunter"
    values = services(pf)
    qualify_services(pf, values)
    values[service][field] = value
    with pytest.raises(ResetError, match="SCM contract"):
        qualify_services(pf, values)


@pytest.mark.parametrize(
    "service,dependencies",
    [
        ("CryptoHunterBackend", ["WrongService"]),
        ("CryptoHunterBackend", ["CryptoHunterPostgreSQL", "ExtraService"]),
        ("CryptoHunterFreshnessVerifier", ["CryptoHunterPostgreSQL"]),
        ("CryptoHunterPostgreSQL", ["RpcSs"]),
    ],
)
def test_service_contract_rejects_wrong_or_extra_dependencies(tmp_path, service, dependencies):
    pf = tmp_path / "CryptoHunter"
    values = services(pf)
    values[service]["Dependencies"] = dependencies
    with pytest.raises(ResetError, match="SCM contract"):
        qualify_services(pf, values)


def test_query_services_reads_backend_dependency_from_service_controller(monkeypatch):
    observed = {}
    payload = [
        {
            "Name": "CryptoHunterBackend",
            "PathName": r'"C:\Program Files\CryptoHunter\CryptoHunterBackend.exe"',
            "StartMode": "Manual",
            "State": "Stopped",
            "StartName": r"NT SERVICE\CryptoHunterBackend",
            "Dependencies": ["CryptoHunterPostgreSQL"],
        }
    ]

    def completed(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(stdout=json.dumps(payload))

    monkeypatch.setattr(subprocess, "run", completed)

    assert query_services()["CryptoHunterBackend"]["Dependencies"] == ["CryptoHunterPostgreSQL"]
    script = observed["command"][-1]
    assert "Get-CimInstance Win32_Service" in script
    assert "Get-Service -Name $service.Name -ErrorAction Stop" in script
    assert "ServicesDependedOn" in script
    assert "ForEach-Object {$_.Name}" in script
    assert "Select-Object Name,PathName,StartMode,State,StartName,Dependencies" not in script
    assert observed["kwargs"]["check"] is True


def test_service_contract_accepts_only_backend_to_postgresql_dependency(tmp_path):
    pf = tmp_path / "CryptoHunter"

    qualify_services(pf, services(pf))


def test_scm_dependency_read_failure_prevents_uninstall_and_purge(tmp_path, monkeypatch):
    _, pd, msi, manifest = runnable(tmp_path, monkeypatch)
    mutations = []

    def dependency_read_failure():
        raise subprocess.CalledProcessError(1, ["powershell.exe"])

    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.query_services",
        dependency_read_failure,
    )
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.uninstall",
        lambda *_: mutations.append("uninstall"),
    )
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.guarded_purge",
        lambda *_: mutations.append("purge"),
    )

    with pytest.raises(subprocess.CalledProcessError):
        run(args(tmp_path, msi, manifest, execute=True))
    assert mutations == []
    assert pd.exists()


def test_cli_has_no_force_or_bypass(capsys):
    with pytest.raises(SystemExit):
        main(["--force"])
    help_text = capsys.readouterr().err
    assert "unrecognized arguments" not in help_text  # required args fail first
    source = Path("deployment/windows_stage9_pre_enrollment_reset.py").read_text()
    assert 'add_argument("--force"' not in source
    assert 'add_argument("--bypass"' not in source
    assert "CryptoHunter-Production-Authority" not in source


def runnable(tmp_path, monkeypatch):
    pf, pd = machine(tmp_path, monkeypatch)
    msi, manifest, files = artifact(tmp_path)
    for name in files:
        path = pf / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        files[name] = _sha(path)
    value = json.loads(manifest.read_text())
    value["installed_files"] = files
    value["installed_payload_sha256"] = installed_payload_digest(files)
    manifest.write_text(json.dumps(value))
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.require_host", lambda **_: None
    )
    return pf, pd, msi, manifest


class WindowsTestPath(type(Path())):
    """A Windows-looking path whose resolution can be exercised on Linux."""

    def resolve(self, *args, **kwargs):
        return self


def args(tmp_path, msi, manifest, *, execute):
    return argparse.Namespace(
        installed_msi=msi,
        installed_manifest=manifest,
        receipt=WindowsTestPath(r"C:\safe\receipt.json"),
        execute=execute,
        _receipt_value=None,
        _qualified_receipt=None,
    )


@pytest.mark.parametrize("owned_tree", ["ProgramData", "Program Files"])
@pytest.mark.parametrize("execute", [False, True])
def test_receipt_in_product_tree_fails_before_any_work(
    tmp_path, monkeypatch, capsys, owned_tree, execute
):
    pf, pd, msi, manifest = runnable(tmp_path, monkeypatch)
    root = pd if owned_tree == "ProgramData" else pf
    receipt = root / "evidence/reset.json"
    called = []
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.require_host",
        lambda **_: called.append("host"),
    )
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset._product_code",
        lambda _: called.append("product-code"),
    )
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.uninstall",
        lambda *_: called.append("uninstall"),
    )
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.guarded_purge",
        lambda *_: called.append("purge"),
    )

    result = main(
        [
            "--installed-msi",
            str(msi),
            "--installed-manifest",
            str(manifest),
            "--receipt",
            str(receipt),
            *(["--execute"] if execute else []),
        ]
    )

    assert result == 1
    assert called == []
    assert not receipt.parent.exists()
    assert "PRE_ENROLLMENT_RESET = FAIL (ResetError)" in capsys.readouterr().err


def test_external_receipt_and_derived_log_are_outside_product_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("ProgramFiles", r"D:\Program Files")
    monkeypatch.setenv("ProgramData", r"D:\ProgramData")
    receipt, log = qualify_output_path(WindowsTestPath(r"C:\evidence\pre-enrollment-reset.json"))
    assert receipt == Path(r"C:\evidence\pre-enrollment-reset.json")
    assert log == Path(r"C:\evidence\pre-enrollment-reset.uninstall.log")


@pytest.mark.parametrize(
    "receipt",
    [
        r"\\?\C:\ProgramData\CryptoHunter\receipt.json",
        r"\\?\C:\evidence\receipt.json",
        r"\\.\C:\ProgramData\CryptoHunter\receipt.json",
        r"\\localhost\C$\ProgramData\CryptoHunter\receipt.json",
        r"\\server\share\receipt.json",
        r"\\?\UNC\server\share\receipt.json",
        r"C:evidence\receipt.json",
    ],
)
def test_non_local_output_namespace_fails_before_any_work(tmp_path, monkeypatch, capsys, receipt):
    _, _, msi, manifest = runnable(tmp_path, monkeypatch)
    called = []
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.Path.resolve",
        lambda self: called.append("resolve") or self,
    )
    for name in (
        "require_host",
        "_product_code",
        "uninstall",
        "guarded_purge",
        "_write_receipt",
    ):
        monkeypatch.setattr(
            f"deployment.windows_stage9_pre_enrollment_reset.{name}",
            lambda *_, _name=name, **__: called.append(_name),
        )

    result = main(
        [
            "--installed-msi",
            str(msi),
            "--installed-manifest",
            str(manifest),
            "--receipt",
            receipt,
        ]
    )

    assert result == 1
    assert called == []
    assert "PRE_ENROLLMENT_RESET = FAIL (ResetError)" in capsys.readouterr().err


def test_relative_receipt_resolved_to_local_dos_path_passes(monkeypatch):
    monkeypatch.setenv("ProgramFiles", r"D:\Program Files")
    monkeypatch.setenv("ProgramData", r"D:\ProgramData")
    original_resolve = Path.resolve

    def resolve(path):
        if path == Path(r"evidence\receipt.json"):
            return Path(r"C:\safe\evidence\receipt.json")
        return original_resolve(path)

    monkeypatch.setattr("deployment.windows_stage9_pre_enrollment_reset.Path.resolve", resolve)
    receipt, log = qualify_output_path(Path(r"evidence\receipt.json"))
    assert receipt == Path(r"C:\safe\evidence\receipt.json")
    assert log == Path(r"C:\safe\evidence\receipt.uninstall.log")


@pytest.mark.parametrize(
    "resolved",
    [
        r"\\server\share\receipt.json",
        r"\\?\C:\safe\receipt.json",
        r"\\.\C:\safe\receipt.json",
    ],
)
def test_relative_receipt_resolved_to_non_local_namespace_fails(monkeypatch, resolved):
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.Path.resolve",
        lambda self: Path(resolved),
    )
    with pytest.raises(ResetError, match="local DOS path"):
        qualify_output_path(Path(r"evidence\receipt.json"))


@pytest.mark.parametrize(
    "receipt",
    [
        r"C:\ProgramData\CryptoHunter:reset.json",
        r"C:\Program Files\CryptoHunter:reset.json",
        r"C:\evidence\reset.json:stream",
        r"C:\evidence\reset.json::$DATA",
        r"C:\evidence:stream\reset.json",
    ],
)
@pytest.mark.parametrize("execute", [False, True])
def test_ntfs_stream_receipt_fails_before_any_work(tmp_path, monkeypatch, capsys, receipt, execute):
    _, _, msi, manifest = runnable(tmp_path, monkeypatch)
    called = []
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.Path.resolve",
        lambda self: called.append("resolve") or self,
    )
    for name in (
        "require_host",
        "_product_code",
        "uninstall",
        "guarded_purge",
        "_write_receipt",
    ):
        monkeypatch.setattr(
            f"deployment.windows_stage9_pre_enrollment_reset.{name}",
            lambda *_, _name=name, **__: called.append(_name),
        )

    result = main(
        [
            "--installed-msi",
            str(msi),
            "--installed-manifest",
            str(manifest),
            "--receipt",
            receipt,
            *(["--execute"] if execute else []),
        ]
    )

    assert result == 1
    assert called == []
    assert "PRE_ENROLLMENT_RESET = FAIL (ResetError)" in capsys.readouterr().err


def test_normal_windows_receipt_path_is_not_an_ntfs_stream(monkeypatch):
    monkeypatch.setenv("ProgramFiles", r"D:\Program Files")
    monkeypatch.setenv("ProgramData", r"D:\ProgramData")
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.Path.resolve", lambda self: self
    )
    receipt, log = qualify_output_path(Path(r"C:\evidence\pre-enrollment-reset.json"))
    assert receipt == Path(r"C:\evidence\pre-enrollment-reset.json")
    assert log == Path(r"C:\evidence\pre-enrollment-reset.uninstall.log")


@pytest.mark.parametrize("state", [-1, 2, 3, 5, 999])
def test_product_registration_requires_unknown(monkeypatch, state):
    msi = SimpleNamespace(MsiQueryProductStateW=lambda _: state)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(msi=msi), raising=False)
    if state == -1:
        require_product_unregistered("code")
    else:
        with pytest.raises(ResetError, match=rf"registered \(state {state}\)"):
            require_product_unregistered("code")


def test_valid_dry_run_never_uninstalls_or_deletes(tmp_path, monkeypatch, capsys):
    pf, pd, msi, manifest = runnable(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.query_services",
        lambda: services(pf),
    )
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.uninstall",
        lambda *_: pytest.fail("dry-run invoked uninstall"),
    )
    receipt = run(args(tmp_path, msi, manifest, execute=False))
    assert receipt["result"] == "PASS"
    assert pd.exists()
    output = capsys.readouterr().out
    assert "RESET_ELIGIBLE = YES" in output
    assert "MUTATION = NOT_PERFORMED" in output


def test_uninstall_failure_preserves_program_data(tmp_path, monkeypatch):
    pf, pd, msi, manifest = runnable(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.query_services",
        lambda: services(pf),
    )
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset._product_code", lambda _: "code"
    )
    monkeypatch.setattr("deployment.windows_stage9_pre_enrollment_reset.uninstall", lambda *_: 1603)
    with pytest.raises(ResetError, match="1603"):
        run(args(tmp_path, msi, manifest, execute=True))
    assert pd.exists()


def test_toctou_authority_marker_prevents_purge(tmp_path, monkeypatch):
    pf, pd, msi, manifest = runnable(tmp_path, monkeypatch)
    calls = iter([services(pf), {}])
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.query_services",
        lambda: next(calls),
    )
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset._product_code", lambda _: "code"
    )
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.require_product_unregistered",
        lambda _: None,
    )

    def uninstall_with_race(*_):
        import shutil

        shutil.rmtree(pf)
        (pd / "State/corehost.sqlite").touch()
        return 0

    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.uninstall", uninstall_with_race
    )
    with pytest.raises(ResetError, match="authority/enrollment"):
        run(args(tmp_path, msi, manifest, execute=True))
    assert pd.exists()


def test_msi_absent_state_fails_closed_before_program_data_purge(tmp_path, monkeypatch):
    pf, pd, msi, manifest = runnable(tmp_path, monkeypatch)
    calls = iter([services(pf), {}])
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.query_services",
        lambda: next(calls),
    )
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset._product_code", lambda _: "code"
    )
    msi_api = SimpleNamespace(MsiQueryProductStateW=lambda _: 2)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(msi=msi_api), raising=False)

    def normal_uninstall(*_):
        import shutil

        shutil.rmtree(pf)
        return 0

    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.uninstall", normal_uninstall
    )
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.guarded_purge",
        lambda *_: pytest.fail("MSI state 2 allowed ProgramData purge"),
    )

    with pytest.raises(ResetError, match=r"registered \(state 2\)"):
        run(args(tmp_path, msi, manifest, execute=True))
    assert pd.exists()


def test_successful_execute_reaches_clean_target(tmp_path, monkeypatch):
    pf, pd, msi, manifest = runnable(tmp_path, monkeypatch)
    calls = iter([services(pf), {}, {}])
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.query_services",
        lambda: next(calls),
    )
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset._product_code", lambda _: "code"
    )
    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.require_product_unregistered",
        lambda _: None,
    )

    def normal_uninstall(*_):
        import shutil

        shutil.rmtree(pf)
        return 0

    monkeypatch.setattr(
        "deployment.windows_stage9_pre_enrollment_reset.uninstall", normal_uninstall
    )
    receipt = run(args(tmp_path, msi, manifest, execute=True))
    assert receipt["result"] == "PASS"
    assert receipt["final_clean_preconditions"] == "PASS"
    assert not pf.exists() and not pd.exists()
