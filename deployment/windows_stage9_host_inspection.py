"""Read-only PRE-01B host inspection and passive post-install verification.

Saved observations are rehearsal inputs only. Only the built-in collector can
produce live observations; no approval or assertion dictionary authorizes an
installation, service start, enrollment, or Stage 10. Missing access fails closed.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import os
import re
import stat
import subprocess
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, TypeGuard

from deployment import windows_stage9_postgresql_passive as passive_sql
from deployment.windows_installer.contract import CONTRACT
from deployment.windows_postgresql_auth_contract import (
    final_hba_lines,
    ident_lines,
    qualify_hba_rows,
)
from deployment.windows_stage10_runtime_provenance import qualify_runtime_provenance

SERVICES = (CONTRACT.postgresql_service, CONTRACT.backend_service, CONTRACT.verifier_service)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REPARSE = 0x400


class HostInspectionError(ValueError):
    """A required read or a binding cannot be established."""


# No Win32_Product: enumerating that provider can trigger MSI repair. Known
# folders come from the native 64-bit OS, not operator-supplied product roots.
_HOST_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$pf = Join-Path ([Environment]::GetFolderPath('ProgramFiles')) 'CryptoHunter'
$pd = Join-Path ([Environment]::GetFolderPath('CommonApplicationData')) 'CryptoHunter'
$computer = Get-CimInstance Win32_ComputerSystem
$hardware = Get-CimInstance Win32_ComputerSystemProduct
$allServices = @(Get-CimInstance Win32_Service)
$names = @('CryptoHunterPostgreSQL','CryptoHunterBackend','CryptoHunterFreshnessVerifier')
$services = @($allServices | Where-Object { $_.Name -cin $names } | ForEach-Object {
  $s = $_
  $p = if ($s.ProcessId -gt 0) { Get-CimInstance Win32_Process -Filter "ProcessId=$($s.ProcessId)" }
  @{name=$s.Name; account=$s.StartName; command_line=$s.PathName; state=$s.State;
    process_id=[int]$s.ProcessId; process_path=$p.ExecutablePath;
    process_started=if ($p) { $p.CreationDate.ToUniversalTime().ToString('o') } else { $null }}
})
$products = @()
$views = @([Microsoft.Win32.RegistryView]::Registry64,
  [Microsoft.Win32.RegistryView]::Registry32)
foreach ($view in $views) {
  $base = [Microsoft.Win32.RegistryKey]::OpenBaseKey(
    [Microsoft.Win32.RegistryHive]::LocalMachine,$view)
  try {
    $key = $base.OpenSubKey('SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall')
    if ($null -eq $key) { throw 'uninstall registry unavailable' }
    try {
      foreach ($name in $key.GetSubKeyNames()) {
        $product = $key.OpenSubKey($name)
        if ($null -eq $product) { throw 'uninstall registration unavailable' }
        try {
          if ([string]$product.GetValue('DisplayName') -like '*CryptoHunter*') {
            $products += @{product_code=$name; name=$product.GetValue('DisplayName');
              version=$product.GetValue('DisplayVersion');
              windows_installer=$product.GetValue('WindowsInstaller');
              install_location=$product.GetValue('InstallLocation'); view=$view.ToString()}
          }
        } finally { $product.Dispose() }
      }
    } finally { $key.Dispose() }
  } finally { $base.Dispose() }
}
$enrollment = @()
foreach ($relative in @('State\corehost.sqlite','State\enrollment-finalization.json')) {
  $path = Join-Path $pd $relative
  if (Test-Path -LiteralPath $path -ErrorAction Stop) { $enrollment += $relative }
}
$residue = @(Get-ChildItem -LiteralPath (Split-Path $pd) -Force | Where-Object {
  $_.Name -like '.CryptoHunter.stage9*'
} | ForEach-Object Name)
$conflicts = @($allServices | Where-Object {
  ($_.Name -cnotin $names) -and (($_.Name -like '*CryptoHunter*') -or ($_.PathName -like "*$pf*"))
} | ForEach-Object Name)
$listeners = @(Get-NetTCPConnection -State Listen -ErrorAction Stop |
  Where-Object LocalPort -EQ 55432 |
  ForEach-Object {
    $p = Get-CimInstance Win32_Process -Filter "ProcessId=$($_.OwningProcess)"
    @{address=$_.LocalAddress; port=[int]$_.LocalPort; process_id=[int]$_.OwningProcess;
      process_path=$p.ExecutablePath; parent_process_id=[int]$p.ParentProcessId}
  })
@{
  os='Windows'; native_x64=([Environment]::Is64BitProcess -and
    [Environment]::Is64BitOperatingSystem -and
    $env:PROCESSOR_ARCHITECTURE -ceq 'AMD64' -and
    @((Get-CimInstance Win32_Processor) | Where-Object Architecture -NE 9).Count -eq 0);
  host_identity=@{hostname=$computer.Name;
    machine_guid=(Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Cryptography').MachineGuid;
    hardware_uuid=$hardware.UUID};
  paths=@{program_files=$pf; program_data=$pd;
    program_files_exists=(Test-Path -LiteralPath $pf -ErrorAction Stop);
    program_data_exists=(Test-Path -LiteralPath $pd -ErrorAction Stop)};
  products=$products; services=$services;
  enrollment=@{state=if ($enrollment.Count) {'present'} else {'absent'};
    files=$enrollment}; conflicts=$conflicts; residue=$residue; listeners=$listeners
} | ConvertTo-Json -Depth 10 -Compress
"""

_FINAL_SERVICE_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$names = @('CryptoHunterPostgreSQL','CryptoHunterBackend','CryptoHunterFreshnessVerifier')
@(Get-CimInstance Win32_Service | Where-Object { $_.Name -cin $names } | ForEach-Object {
  $s = $_
  $p = if ($s.ProcessId -gt 0) { Get-CimInstance Win32_Process -Filter "ProcessId=$($s.ProcessId)" }
  @{name=$s.Name; state=$s.State; process_id=[int]$s.ProcessId; command_line=$s.PathName;
    process_path=$p.ExecutablePath;
    process_started=if ($p) { $p.CreationDate.ToUniversalTime().ToString('o') } else { $null }}
}) | ConvertTo-Json -Depth 4 -Compress
"""


def _powershell(script: str, *, env: dict[str, str] | None = None) -> Any:
    from deployment.windows_runner_acl import native_powershell_path

    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    result = subprocess.run(
        [native_powershell_path(), "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    return json.loads(result.stdout.lstrip("\ufeff"))


def _service_configuration(name: str) -> dict[str, Any]:
    scm = importlib.import_module("win32service")
    manager = scm.OpenSCManager(None, None, scm.SC_MANAGER_CONNECT)
    service = None
    try:
        service = scm.OpenService(
            manager, name, scm.SERVICE_QUERY_CONFIG | scm.SERVICE_QUERY_STATUS
        )
        config = scm.QueryServiceConfig(service)
        status = scm.QueryServiceStatusEx(service)
        recovery = scm.QueryServiceConfig2(service, scm.SERVICE_CONFIG_FAILURE_ACTIONS)
        return {
            "service_type": config[0],
            "start_type": config[1],
            "dependencies": list(config[6]),
            "recovery": recovery,
            "second_process_id": status["ProcessId"],
            "second_state": status["CurrentState"],
        }
    finally:
        if service is not None:
            scm.CloseServiceHandle(service)
        scm.CloseServiceHandle(manager)


def collect_host_observation() -> dict[str, Any]:
    """Collect native reads without opening a service for modification."""
    if os.name != "nt":
        raise HostInspectionError("native Windows x64 host access is required")
    value = _powershell(_HOST_SCRIPT)
    if not isinstance(value, dict) or not isinstance(value.get("services"), list):
        raise HostInspectionError("host collector returned malformed data")
    value["related_products"] = _related_products()
    value["all_user_msi"] = _all_user_msi_inventory()
    for service in value["services"]:
        service.update(_service_configuration(service["name"]))
    return value


def _related_products() -> list[dict[str, Any]]:
    """Enumerate the frozen MSI upgrade family independently of DisplayName."""
    client = importlib.import_module("win32com.client")
    installer = client.Dispatch("WindowsInstaller.Installer")
    return [
        {
            "product_code": str(code),
            "version": str(installer.ProductInfo(code, "VersionString")),
            "state": installer.ProductState(code),
        }
        for code in installer.RelatedProducts("{" + CONTRACT.upgrade_code + "}")
    ]


def _all_user_msi_inventory() -> dict[str, Any]:
    """Read all user/machine MSI contexts; inaccessible caches stay unknown.

    RelatedProducts alone covers the current user and machine, so cannot prove
    the absence of another user's deployment. ProductsEx with Everyone and all
    three contexts requires the native elevated read authority. Every product's
    cached MSI must be available and stable before its upgrade family is known.
    """
    client = importlib.import_module("win32com.client")
    installer = client.Dispatch("WindowsInstaller.Installer")
    family = []
    conflicts = []
    for count, product in enumerate(installer.ProductsEx("", "S-1-1-0", 7), 1):
        if count > 4096:
            raise HostInspectionError("complete all-user MSI inventory exceeds the read bound")
        code = str(product.ProductCode)
        context = product.Context
        user_sid = str(product.UserSid)
        install_state = str(product.InstallProperty("State"))
        cache = Path(str(product.InstallProperty("LocalPackage")))
        if (
            install_state != "5"
            or re.fullmatch(r"\{[0-9A-Fa-f-]{36}\}", code) is None
            or type(context) is not int
            or context not in (1, 2, 4)
            or (context != 4 and re.fullmatch(r"S-1-5-21-(?:\d+-){3}\d+", user_sid) is None)
            or not cache.is_absolute()
        ):
            raise HostInspectionError("all-user MSI context/cache is unavailable")
        digest = _sha(cache)
        cached_code = _msi_property(installer, cache, "ProductCode")
        upgrade = _msi_property(installer, cache, "UpgradeCode")
        name = _msi_property(installer, cache, "ProductName")
        version = _msi_property(installer, cache, "ProductVersion")
        if (
            not code
            or not cached_code
            or cached_code.upper() != code.upper()
            or not name
            or not version
            or _sha(cache) != digest
        ):
            raise HostInspectionError("all-user MSI cached identity is unavailable or changed")
        record = {
            "product_code": code,
            "version": version,
            "context": context,
            "user_sid": user_sid,
        }
        if upgrade and upgrade.strip("{}").upper() == CONTRACT.upgrade_code:
            family.append(record)
        elif "cryptohunter" in name.casefold():
            conflicts.append(record)
    return {
        "complete": True,
        "user_sid": "S-1-1-0",
        "context_mask": 7,
        "family_products": family,
        "conflicting_products": conflicts,
    }


def _all_user_inventory_complete(value: Any) -> TypeGuard[dict[str, Any]]:
    return (
        isinstance(value, dict)
        and value.get("complete") is True
        and value.get("user_sid") == "S-1-1-0"
        and value.get("context_mask") == 7
        and isinstance(value.get("family_products"), list)
        and isinstance(value.get("conflicting_products"), list)
    )


def _report(kind: str, live: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "report_type": kind,
        "evidence_mode": "live" if live else "rehearsal",
        "stage_9": "IN_PROGRESS",
        "legal_production_enrollment": "NOT_PERFORMED",
        "run_stage10": False,
        "stage10": "BLOCKED / NOT_RUN",
        "install_execution_authorized": False,
        "checks": [],
    }


def _check(report: dict[str, Any], name: str, status: str, details: str) -> None:
    if status == "PASS" and report["evidence_mode"] != "live":
        status = "UNVERIFIED"
        details = "Rehearsal consistency only; host was not reached. " + details
    report["checks"].append({"name": name, "status": status, "details": details})


def _finish(report: dict[str, Any]) -> dict[str, Any]:
    statuses = {item["status"] for item in report["checks"]}
    report["status"] = report["overall_status"] = next(
        (value for value in ("FAIL", "BLOCKED", "UNVERIFIED") if value in statuses),
        "PASS" if statuses else "BLOCKED",
    )
    return report


def _identity(value: Any) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("hostname"), str):
        return False
    try:
        return bool(value["hostname"].strip()) and all(
            uuid.UUID(value[key]).int != 0 for key in ("machine_guid", "hardware_uuid")
        )
    except (KeyError, ValueError, TypeError, AttributeError):
        return False


def _paths(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    for key in ("program_files", "program_data"):
        path = value.get(key)
        if (
            not isinstance(path, str)
            or re.fullmatch(r"[A-Za-z]:\\[^\r\n]+", path) is None
            or any(part in {".", "..", ""} for part in path.split("\\")[1:])
            or PureWindowsPath(path).name != CONTRACT.product_name
            or type(value.get(key + "_exists")) is not bool
        ):
            return False
    return PureWindowsPath(value["program_files"]) != PureWindowsPath(value["program_data"])


def _inspect(observation: Any, *, live: bool) -> dict[str, Any]:
    report = _report("windows-stage9-existing-installation", live)
    if not isinstance(observation, dict):
        _check(report, "host_access", "BLOCKED", "Host observations are unavailable.")
        return _finish(report)
    report["host_identity"] = observation.get("host_identity")
    # Retain the read-only diagnostic facts, including version and service
    # arguments, so an operator can review conflicts without automatic repair.
    report["observation"] = observation
    for name, valid in (
        (
            "native_windows_x64",
            observation.get("os") == "Windows" and observation.get("native_x64") is True,
        ),
        ("host_identity", _identity(observation.get("host_identity"))),
        ("canonical_product_paths", _paths(observation.get("paths"))),
    ):
        _check(
            report, name, "PASS" if valid else "BLOCKED", "Required native host identity/path read."
        )
    enrollment = observation.get("enrollment")
    absent = (
        isinstance(enrollment, dict)
        and enrollment.get("state") == "absent"
        and enrollment.get("files") == []
    )
    _check(
        report,
        "enrollment_absent",
        "PASS" if absent else "BLOCKED",
        "Existing or unknown enrollment requires a separate operator decision.",
    )
    paths = observation.get("paths", {})
    if not isinstance(paths, dict):
        paths = {}
    for name in ("products", "related_products", "services", "conflicts", "residue", "listeners"):
        value = observation.get(name)
        _check(
            report,
            name + "_absent",
            "PASS" if isinstance(value, list) and not value else "BLOCKED",
            "Existing installation, conflicting endpoint or residue is never taken over, "
            "reset or repaired.",
        )
    inventory = observation.get("all_user_msi")
    no_user_product = (
        _all_user_inventory_complete(inventory)
        and inventory["family_products"] == []
        and inventory["conflicting_products"] == []
    )
    _check(
        report,
        "all_user_msi_absent",
        "PASS" if no_user_product else "BLOCKED",
        "Complete all-user installed-context MSI enumeration and readable cached upgrade "
        "identities are required; missing access/advertised cache or another user's "
        "deployment stays blocked.",
    )
    clean_paths = (
        paths.get("program_files_exists") is False and paths.get("program_data_exists") is False
    )
    _check(
        report,
        "product_directories_absent",
        "PASS" if clean_paths else "BLOCKED",
        "Both canonical product directories must be absent for a fresh installation.",
    )
    return _finish(report)


def inspect_existing_installation(observation: Any) -> dict[str, Any]:
    """Validate saved observations, without treating them as live evidence."""
    return _inspect(observation, live=False)


def run_host_inspection(*, live: bool = False, snapshot: Any = None) -> dict[str, Any]:
    """Only live mode invokes the native collector; snapshots cannot select it."""
    if live and snapshot is not None:
        report = _report("windows-stage9-existing-installation", True)
        _check(report, "live_collection", "FAIL", "Live inspection cannot consume a snapshot.")
        return _finish(report)
    try:
        return _inspect(collect_host_observation() if live else snapshot, live=live)
    except Exception as exc:
        report = _report("windows-stage9-existing-installation", live)
        _check(
            report,
            "host_access",
            "BLOCKED",
            f"Read-only host collection failed ({type(exc).__name__}).",
        )
        return _finish(report)


def _sha(path: Path) -> str:
    absolute = path.absolute()
    before = {}
    for candidate in (absolute, *absolute.parents):
        metadata = candidate.lstat()
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & _REPARSE:
            raise HostInspectionError("qualified file path contains a redirected component")
        before[candidate] = (metadata.st_dev, metadata.st_ino)
    if not stat.S_ISREG(absolute.lstat().st_mode):
        raise HostInspectionError("required qualified file is absent")
    with absolute.open("rb") as stream:
        start = os.fstat(stream.fileno())
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        end = os.fstat(stream.fileno())
    for candidate, identity in before.items():
        metadata = candidate.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or getattr(metadata, "st_file_attributes", 0) & _REPARSE
            or (metadata.st_dev, metadata.st_ino) != identity
        ):
            raise HostInspectionError("qualified file path changed during reading")
    latest = absolute.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(
        getattr(start, field) != getattr(end, field)
        or getattr(end, field) != getattr(latest, field)
        for field in fields
    ):
        raise HostInspectionError("qualified file changed during reading")
    return digest


def _installed_files(root: Path, expected: Any) -> None:
    if not isinstance(expected, dict) or not expected:
        raise HostInspectionError("installed_files must be a nonempty hash mapping")
    for name, digest in expected.items():
        relative = PurePosixPath(name) if isinstance(name, str) else None
        if (
            relative is None
            or relative.is_absolute()
            or relative.as_posix() != name
            or "\\" in name
            or ":" in name
            or ".." in relative.parts
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise HostInspectionError("unsafe installed_files path or digest")
    before: dict[str, tuple[int, int, int, int]] = {}
    observed: dict[str, str] = {}
    for path in (root, *sorted(root.rglob("*"))):
        metadata = path.lstat()
        if path.is_symlink() or getattr(metadata, "st_file_attributes", 0) & _REPARSE:
            raise HostInspectionError("installed payload contains a redirected path")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            before[name] = (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
            )
            observed[name] = _sha(path)
    after = {
        path.relative_to(root).as_posix(): (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
        for path in root.rglob("*")
        if (metadata := path.lstat()) and stat.S_ISREG(metadata.st_mode)
    }
    if before != after or observed != expected:
        raise HostInspectionError("installed file set/SHA-256 differs or changed during inspection")


def _msi_property(installer: Any, msi: Path, property_name: str) -> str | None:
    if property_name not in {"ProductCode", "UpgradeCode", "ProductName", "ProductVersion"}:
        raise HostInspectionError("unsupported MSI identity property")
    database = installer.OpenDatabase(str(msi.resolve()), 0)
    view = database.OpenView(f"SELECT `Value` FROM `Property` WHERE `Property` = '{property_name}'")
    try:
        view.Execute(None)
        record = view.Fetch()
        return str(record.StringData(1)) if record is not None else None
    finally:
        view.Close()


def _msi_product_code(msi: Path) -> str:
    """Read the qualified MSI database in read-only mode; never install/repair."""
    client = importlib.import_module("win32com.client")
    installer = client.Dispatch("WindowsInstaller.Installer")
    code = _msi_property(installer, msi, "ProductCode")
    if not code:
        raise HostInspectionError("qualified MSI ProductCode is absent")
    return code


def _collect_post_install(
    msi: Path, *, include_readiness: bool = True, include_configuration: bool = False
) -> dict[str, Any]:
    observation = collect_host_observation()
    observation["qualified_product_code"] = _msi_product_code(msi)
    paths = observation["paths"]
    root, machine = Path(paths["program_files"]), Path(paths["program_data"])
    data = machine / "PostgreSQL" / "Data"
    observation["postgresql"] = {
        "hba_lines": (data / "pg_hba.conf").read_text(encoding="utf-8").splitlines(),
        "ident_lines": (data / "pg_ident.conf").read_text(encoding="utf-8").splitlines(),
    }
    if include_configuration:
        for name in ("postgresql.conf", "postgresql.auto.conf"):
            path = data / name
            digest = _sha(path)
            lines = path.read_text(encoding="utf-8").splitlines()
            if _sha(path) != digest:
                raise HostInspectionError("PostgreSQL static configuration changed during reading")
            observation["postgresql"][name] = lines
    readiness = {}
    for service, relative in (
        (
            (CONTRACT.backend_service, "backend-readiness.json"),
            (CONTRACT.verifier_service, "Verifier/verifier-readiness.json"),
        )
        if include_readiness
        else ()
    ):
        path = machine / "Runtime" / relative
        if path.exists():
            readiness[service] = {
                "marker": json.loads(path.read_text(encoding="utf-8")),
                "written": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(),
            }
    observation["readiness"] = readiness
    # Reuse the exact existing product DACL contract, never provision.qualify()
    # (which calls the recovery setter) or qualify_mtls() (which starts services).
    provision = importlib.import_module("deployment.windows_installer.provision")
    provision.qualify_dacl(root, machine)
    observation["product_dacl"] = "PASS"
    from deployment.windows_runner_acl import WINDOWS_ACL_TREE_SCRIPT

    observation["acl_trees"] = {}
    for name, path in (("program_files", root), ("program_data", machine)):
        env = dict(os.environ)
        env["CRYPTOHUNTER_INSPECTION_ROOT"] = str(path)
        observation["acl_trees"][name] = _powershell(
            WINDOWS_ACL_TREE_SCRIPT
            + "\nCollect-RunnerAclTree -Root $env:CRYPTOHUNTER_INSPECTION_ROOT "
            "| ConvertTo-Json -Depth 16 -Compress",
            env=env,
        )
    security = importlib.import_module("win32security")
    observation["service_sids"] = [
        security.ConvertSidToStringSid(security.LookupAccountName(None, rf"NT SERVICE\{name}")[0])
        for name in SERVICES
    ]
    # This process may read only its own service credentials, after independent
    # native token/SCM checks. An administrator inspecting a clean base receives
    # an opaque BLOCKED receipt without touching either client's private key.
    observation["postgresql"]["effective_sql"] = passive_sql.collect_passive_postgresql(observation)
    return observation


def _services(
    observation: dict[str, Any], root: str, *, enrolled: bool = False, clean_base: bool = False
) -> None:
    values = observation.get("services")
    if not isinstance(values, list) or len(values) != len(SERVICES):
        raise HostInspectionError("exactly the three canonical services are required")
    services = {item["name"]: item for item in values}
    if set(services) != set(SERVICES):
        raise HostInspectionError("canonical service identities differ")
    for name in SERVICES:
        value = services[name]
        binary = (
            PureWindowsPath(root)
            / ("PostgreSQL" if name == CONTRACT.postgresql_service else "")
            / f"{name}.exe"
        )
        command = value.get("command_line")
        if (
            command
            not in (
                f'"{binary}"',
                str(binary) if not any(c.isspace() for c in str(binary)) else None,
            )
            or value.get("account") != rf"NT SERVICE\{name}"
            or value.get("service_type") != 16
            or value.get("start_type")
            != (
                2
                if name == CONTRACT.postgresql_service
                or (enrolled and name == CONTRACT.backend_service)
                else 3
            )
            or value.get("dependencies")
            != ([CONTRACT.postgresql_service] if name == CONTRACT.backend_service else [])
        ):
            raise HostInspectionError(
                f"SCM path/arguments/account/start/dependencies differ: {name}"
            )
        recovery = value.get("recovery", {})
        actions = recovery.get("Actions")
        if recovery.get("Command") not in (None, "") or recovery.get("RebootMsg") not in (None, ""):
            raise HostInspectionError("SCM recovery contains a command or reboot policy")
        if name == CONTRACT.backend_service and not clean_base:
            if (
                recovery.get("ResetPeriod") != 86400
                or not isinstance(actions, (list, tuple))
                or [tuple(action) for action in actions]
                != [(1, 1000), (1, 5000), (1, 30000), (0, 0)]
            ):
                raise HostInspectionError(
                    "backend recovery differs from the existing provision contract"
                )
        elif actions not in ([], None):
            raise HostInspectionError("unexpected service recovery actions")


def _postgresql_static(observation: dict[str, Any]) -> None:
    """Check static auth boundaries and the native private PostgreSQL listener."""
    value = observation["postgresql"]
    if value.get("hba_lines") != list(final_hba_lines("freshness_gate")) or value.get(
        "ident_lines"
    ) != list(ident_lines()):
        raise HostInspectionError("exact PostgreSQL HBA/ident role split differs")
    services = {item["name"]: item for item in observation["services"]}
    service = services[CONTRACT.postgresql_service]
    listeners = observation.get("listeners")
    if (
        service.get("state") != "Running"
        or type(service.get("process_id")) is not int
        or service["process_id"] <= 0
        or service.get("second_process_id") != service["process_id"]
        or service.get("second_state") != 4
        or PureWindowsPath(service.get("process_path", ""))
        != PureWindowsPath(observation["paths"]["program_files"])
        / "PostgreSQL"
        / "CryptoHunterPostgreSQL.exe"
        or not isinstance(listeners, list)
        or len(listeners) != 1
        or listeners[0].get("address") != CONTRACT.postgresql_host
        or listeners[0].get("port") != CONTRACT.postgresql_port
        or type(listeners[0].get("process_id")) is not int
        or listeners[0]["process_id"] <= 0
        or listeners[0].get("parent_process_id") != service["process_id"]
        or PureWindowsPath(listeners[0].get("process_path", ""))
        != PureWindowsPath(observation["paths"]["program_files"])
        / "PostgreSQL"
        / "bin"
        / "postgres.exe"
    ):
        raise HostInspectionError(
            "private PostgreSQL must have only the canonical loopback listener"
        )


def _postgresql_effective(observation: dict[str, Any], *, require_live: bool = True) -> None:
    """Require actual service-token SQL evidence in addition to static files."""
    value = observation.get("postgresql")
    effective = value.get("effective_sql") if isinstance(value, dict) else None
    if require_live:
        summary = passive_sql.summarize_passive_postgresql(effective, observation)
        if any(result["status"] == "FAIL" for result in summary["components"].values()):
            raise HostInspectionError("passive PostgreSQL evidence differs")
        for component in passive_sql.COMPONENTS:
            try:
                passive_sql.require_passive_postgresql_component(effective, observation, component)
            except passive_sql.PassivePostgreSQLError as exc:
                suffix = " unavailable" if exc.status != "FAIL" else " differs"
                raise HostInspectionError("passive PostgreSQL " + component + suffix) from exc
        return
    # Retain saved-fact schema checks solely for rehearsal consistency. Public
    # entry points downgrade these to UNVERIFIED; live calls require a receipt.
    if not isinstance(effective, dict):
        raise HostInspectionError(
            "effective service-token PostgreSQL SQL/role evidence is unavailable"
        )
    if (
        effective.get("listen_addresses") != CONTRACT.postgresql_host
        or effective.get("port") != CONTRACT.postgresql_port
        or effective.get("ssl") is not True
        or effective.get("server_version_num") != 170011
    ):
        raise HostInspectionError("effective PostgreSQL endpoint/TLS differs")
    qualify_hba_rows(effective["hba_rows"], "freshness_gate")
    if (
        effective.get("session_roles") != ["freshness_runtime", "freshness_crypto_verifier"]
        or effective.get("cross_roles_denied") is not True
    ):
        raise HostInspectionError("effective PostgreSQL service role separation differs")


def _postgresql_configuration(observation: dict[str, Any]) -> None:
    """Qualify the static endpoint/TLS settings installed by _configure_database.

    This deliberately cannot establish the server's effective configuration:
    service-token SQL qualification remains a separate, required gate.
    """
    value = observation["postgresql"]
    configuration = value.get("postgresql.conf")
    automatic = value.get("postgresql.auto.conf")
    if (
        not isinstance(configuration, list)
        or not isinstance(automatic, list)
        or any(not isinstance(line, str) for line in [*configuration, *automatic])
    ):
        raise HostInspectionError("PostgreSQL static configuration is unavailable")
    if any(line.strip() and not line.lstrip().startswith("#") for line in automatic):
        raise HostInspectionError("PostgreSQL automatic configuration overrides are prohibited")
    security = PureWindowsPath(observation["paths"]["program_data"]) / "Security"
    expected = {
        "listen_addresses": f"'{CONTRACT.postgresql_host}'",
        "port": str(CONTRACT.postgresql_port),
        "ssl": "on",
        "ssl_ca_file": f"'{(security / 'ca.crt').as_posix()}'",
        "ssl_cert_file": f"'{(security / 'server/server.crt').as_posix()}'",
        "ssl_key_file": f"'{(security / 'server/server.key').as_posix()}'",
    }
    observed = {}
    for line in configuration:
        active = line.strip()
        if not active or active.startswith("#"):
            continue
        if re.match(r"include(?:_dir|_if_exists)?\b", active, re.IGNORECASE):
            raise HostInspectionError("PostgreSQL configuration includes are prohibited")
        key, separator, setting = active.partition("=")
        key = key.strip().lower()
        if key in expected:
            if not separator or key in observed or setting.strip() != expected[key]:
                raise HostInspectionError("PostgreSQL static endpoint/TLS differs")
            observed[key] = setting.strip()
    if observed != expected:
        raise HostInspectionError("PostgreSQL static endpoint/TLS settings are incomplete")


def _report_postgresql_evidence(report: dict[str, Any], observation: dict[str, Any]) -> None:
    """Export provenance and capability gaps without making JSON an authority."""
    value = observation.get("postgresql")
    receipt = value.get("effective_sql") if isinstance(value, dict) else None
    summary = passive_sql.summarize_passive_postgresql(receipt, observation)
    report["postgresql_evidence"] = summary
    missing = []
    for name, result in summary["components"].items():
        if result["status"] != "PASS":
            missing.append("passive PostgreSQL " + name)
        if report["evidence_mode"] == "live":
            _check(report, "postgresql_" + name, result["status"], result["detail"])
    if missing:
        report["missing_authorities"] = missing


def _postgresql(observation: dict[str, Any], *, require_live: bool = True) -> None:
    _postgresql_static(observation)
    _postgresql_effective(observation, require_live=require_live)


def _readiness(observation: dict[str, Any], root: str) -> None:
    markers = observation.get("readiness", {})
    for service in observation["services"]:
        name = service["name"]
        if name == CONTRACT.postgresql_service:
            continue
        pid = service.get("process_id")
        marker = markers.get(name, {})
        value = marker.get("marker", {})
        expected_role = (
            "freshness_runtime" if name == CONTRACT.backend_service else "freshness_crypto_verifier"
        )
        started = datetime.fromisoformat(service["process_started"])
        written = datetime.fromisoformat(marker["written"])
        if (
            service.get("state") != "Running"
            or type(pid) is not int
            or pid <= 0
            or service.get("second_process_id") != pid
            or service.get("second_state") != 4
            or PureWindowsPath(service.get("process_path", ""))
            != PureWindowsPath(root) / f"{name}.exe"
            or type(value.get("pid")) is not int
            or value["pid"] != pid
            or value.get("db_role") != expected_role
            or value.get("cross_key_denied") is not True
            or value.get("cross_roles_denied") is not True
            or (name == CONTRACT.backend_service and value.get("corehost_lock") is not True)
            or started.tzinfo is None
            or written.tzinfo is None
            or written < started
        ):
            raise HostInspectionError("readiness is not bound to the actual stable SCM process")


def _stable_service_instances(initial: Any, final: Any) -> None:
    if (
        not isinstance(initial, list)
        or not isinstance(final, list)
        or len(initial) != 3
        or len(final) != 3
    ):
        raise HostInspectionError("final native SCM/process observations are unavailable")
    first = {item["name"]: item for item in initial}
    last = {item["name"]: item for item in final}
    if set(first) != set(last) or set(last) != set(SERVICES):
        raise HostInspectionError("final SCM service identities differ")
    for name in SERVICES:
        for field in ("state", "process_id", "process_started", "process_path", "command_line"):
            if (
                field not in first[name]
                or field not in last[name]
                or first[name][field] != last[name][field]
            ):
                raise HostInspectionError("SCM process instance changed during verification")


def _qualified_install_inputs(
    qualified_msi: Path,
    manifest_path: Path,
    clean_receipt_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_msi_sha256: str,
    expected_receipt_sha256: str,
) -> dict[str, Any]:
    """Reread the same qualified inputs for both passive installation phases."""
    for path, expected in (
        (qualified_msi, expected_msi_sha256),
        (manifest_path, expected_manifest_sha256),
        (clean_receipt_path, expected_receipt_sha256),
    ):
        if (
            not isinstance(expected, str)
            or _SHA256.fullmatch(expected) is None
            or _sha(path) != expected
        ):
            raise HostInspectionError("qualified MSI, manifest or receipt integrity differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest["msi"]["sha256"] != expected_msi_sha256:
        raise HostInspectionError("qualified manifest/MSI identity differs")
    return manifest


def verify_post_install(
    qualified_msi: Path,
    manifest_path: Path,
    clean_receipt_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_msi_sha256: str,
    expected_receipt_sha256: str,
    source_revision: str,
    ci_run_id: str,
    live: bool = False,
    observation: Any = None,
) -> dict[str, Any]:
    """Verify passively; qualified digests must come from the approved handoff.

    This never starts backend/verifier or changes Demand startup. A normal fresh
    MSI remains BLOCKED on runtime readiness pending the separate legal path.
    Passive SQL receipts expose partial capabilities. Both genuine service
    identities and active-authentication authority remain required separately.
    """
    report = _report("windows-stage9-passive-post-install", live)
    if live and observation is not None:
        _check(
            report,
            "live_collection",
            "FAIL",
            "Live verification cannot consume an observation dictionary.",
        )
        return _finish(report)
    try:
        manifest = _qualified_install_inputs(
            qualified_msi,
            manifest_path,
            clean_receipt_path,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_msi_sha256=expected_msi_sha256,
            expected_receipt_sha256=expected_receipt_sha256,
        )
        _check(
            report,
            "qualified_inputs",
            "PASS",
            "All three approved handoff payload digests were reread.",
        )
    except Exception as exc:
        _check(
            report,
            "qualified_inputs",
            "FAIL",
            f"Qualified payload validation failed ({type(exc).__name__}).",
        )
        return _finish(report)
    try:
        facts = _collect_post_install(qualified_msi) if live else observation
        if not isinstance(facts, dict):
            raise HostInspectionError("host observations are unavailable")
        report["host_identity"] = facts.get("host_identity")
        if (
            facts.get("os") != "Windows"
            or facts.get("native_x64") is not True
            or not _identity(facts.get("host_identity"))
            or not _paths(facts.get("paths"))
        ):
            raise HostInspectionError("native canonical host identity/paths are unavailable")
        root = facts["paths"]["program_files"]
        # Hermetic tests may use a local payload root, but cannot obtain live PASS.
        payload_root = Path(root if live else facts.get("payload_root", root))
    except Exception as exc:
        _check(
            report,
            "host_access",
            "BLOCKED",
            f"Read-only host collection failed ({type(exc).__name__}).",
        )
        return _finish(report)
    operations = (
        ("installed_files", lambda: _installed_files(payload_root, manifest["installed_files"])),
        ("msi_registration", lambda: _registration(facts, manifest)),
        (
            "runtime_provenance",
            lambda: qualify_runtime_provenance(
                manifest_path,
                clean_receipt_path,
                payload_root / "CryptoHunterBackend.exe",
                facts["products"][0]["version"],
                source_revision=source_revision,
                ci_run_id=ci_run_id,
                ci_provider="https://github.com",
            ),
        ),
        ("scm_configuration", lambda: _services(facts, root)),
        ("postgresql", lambda: _postgresql(facts, require_live=live)),
        ("product_acl", lambda: _product_acl(facts)),
        ("runtime_readiness", lambda: _readiness(facts, root)),
    )
    for name, operation in operations:
        try:
            operation()
            _check(
                report,
                name,
                "PASS",
                "Existing product contract is consistent with collected facts.",
            )
        except Exception as exc:
            missing = isinstance(exc, (KeyError, OSError)) or (
                isinstance(exc, HostInspectionError) and "unavailable" in str(exc)
            )
            _check(
                report,
                name,
                "BLOCKED" if missing else "FAIL",
                f"Passive verification failed ({type(exc).__name__}).",
            )
    _report_postgresql_evidence(report, facts)
    enrollment = facts.get("enrollment")
    if (
        not isinstance(enrollment, dict)
        or enrollment.get("state") != "absent"
        or enrollment.get("files") != []
    ):
        _check(
            report,
            "enrollment_boundary",
            "BLOCKED",
            "Existing/unknown enrollment needs the separate legal production verification path.",
        )
    for path, expected in (
        (qualified_msi, expected_msi_sha256),
        (manifest_path, expected_manifest_sha256),
        (clean_receipt_path, expected_receipt_sha256),
    ):
        try:
            unchanged = _sha(path) == expected
        except (OSError, HostInspectionError):
            unchanged = False
        if not unchanged:
            _check(
                report,
                "qualified_input_stability",
                "FAIL",
                "A qualified input changed during passive verification.",
            )
            break
    # This is the last native gate, after all payload rehashes as well as ACL
    # and marker reads. Initial SCM reads cannot catch an intervening restart.
    try:
        final_services = _powershell(_FINAL_SERVICE_SCRIPT) if live else facts.get("final_services")
        _stable_service_instances(facts.get("services"), final_services)
        _check(
            report,
            "final_scm_process_identity",
            "PASS",
            "Final native SCM PID/start time/path still match the verified service instances.",
        )
    except Exception as exc:
        _check(
            report,
            "final_scm_process_identity",
            "BLOCKED"
            if isinstance(exc, (OSError, subprocess.SubprocessError))
            or (isinstance(exc, HostInspectionError) and "unavailable" in str(exc))
            else "FAIL",
            f"Final SCM/process observation failed ({type(exc).__name__}).",
        )
    return _finish(report)


def _registration(facts: dict[str, Any], manifest: dict[str, Any]) -> None:
    products = facts.get("products")
    if not isinstance(products, list) or len(products) != 1:
        raise HostInspectionError("exactly one MSI product registration is required")
    product = products[0]
    code = facts.get("qualified_product_code")
    related = facts.get("related_products")
    inventory = facts.get("all_user_msi")
    if (
        not isinstance(code, str)
        or re.fullmatch(r"\{[0-9A-Fa-f-]{36}\}", code) is None
        or product.get("product_code", "").upper() != code.upper()
        or product.get("name") != CONTRACT.product_name
        or product.get("version") != manifest["product_version"]
        or product.get("windows_installer") != 1
        or product.get("view") != "Registry64"
        or not isinstance(related, list)
        or len(related) != 1
        or related[0].get("product_code", "").upper() != code.upper()
        or related[0].get("version") != manifest["product_version"]
        or related[0].get("state") != 5
        or not _all_user_inventory_complete(inventory)
        or inventory["conflicting_products"] != []
        or len(inventory["family_products"]) != 1
        or inventory["family_products"][0].get("context") != 4
        or inventory["family_products"][0].get("product_code", "").upper() != code.upper()
        or inventory["family_products"][0].get("version") != manifest["product_version"]
    ):
        raise HostInspectionError("MSI product identity/version differs from qualified artifact")


def _product_acl(facts: dict[str, Any]) -> None:
    from deployment.windows_runner_acl import validate_acl_tree

    sids = facts.get("service_sids")
    if (
        facts.get("product_dacl") != "PASS"
        or not isinstance(sids, list)
        or len(sids) != 3
        or any(not isinstance(sid, str) or not sid.startswith("S-1-5-80-") for sid in sids)
    ):
        raise HostInspectionError("exact product DACL/service SID qualification is unavailable")
    for name in ("program_files", "program_data"):
        immutable_payload = name == "program_files"
        errors = validate_acl_tree(
            facts["acl_trees"][name],
            runner_root=facts["paths"][name],
            service_sid="S-1-5-18" if immutable_payload else sids[0],
            trusted_write_sids=frozenset() if immutable_payload else frozenset(sids),
        )
        if errors:
            raise HostInspectionError("protected product tree ACL differs")
