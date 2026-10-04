"""Fail-closed removal of an exact, unenrolled Stage-9 base installation.

This is deliberately not a recovery tool.  The only destructive operation it
performs itself is removal of the already-uninstalled product's qualified,
authority-free ProgramData directory.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import shutil
import subprocess
import sys
from typing import Any

from deployment.windows_installer.build import installed_payload_digest
from deployment.windows_installer.contract import CONTRACT

OPERATION = "STAGE9_PRE_ENROLLMENT_BASE_REPLACEMENT_RESET"
SERVICES = (
    CONTRACT.postgresql_service,
    CONTRACT.backend_service,
    CONTRACT.verifier_service,
)
BASE_TOP_LEVEL = {
    ".stage9-install-ownership.json",
    "Config",
    "Logs",
    "PostgreSQL",
    "Runtime",
    "Security",
    "State",
    "Updates",
}
TRANSACTION_JOURNAL = ".CryptoHunter.stage9-transaction.json"
SAFE_FAILURE_DIAGNOSTIC = ".CryptoHunter.stage9-install-failure.json"
STARTUP_FAILURE_DIAGNOSTIC = ".CryptoHunter.postgresql-service-startup.json"
# Reviewed durable markers created or consumed by production trust publication,
# enrollment finalization, and first backend materialization.  Directory
# existence is sufficient: contents/phases are intentionally never interpreted.
AUTHORITY_MARKERS = (
    Path("Config/ProductionTrust"),
    Path("State/corehost.sqlite"),
    Path("State/enrollment-finalization.json"),
    Path("Runtime/backend-readiness.json"),
    Path("Runtime/Verifier/verifier-readiness.json"),
)


class ResetError(RuntimeError):
    pass


def _is_reparse_point(path: Path) -> bool:
    try:
        return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0) & 0x400)
    except OSError as exc:
        raise ResetError(f"cannot qualify path: {path}") from exc


def transaction_path(program_data: Path) -> Path:
    return program_data.parent / TRANSACTION_JOURNAL


def safe_failure_path() -> Path:
    return Path(os.environ["ProgramData"]) / SAFE_FAILURE_DIAGNOSTIC


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_paths() -> tuple[Path, Path]:
    """Resolve the installer-owned locations from Windows known-folder authority."""
    return (
        Path(os.environ["ProgramFiles"]) / CONTRACT.product_name,
        Path(os.environ["ProgramData"]) / CONTRACT.product_name,
    )


def require_host(*, execute: bool) -> None:
    if os.name != "nt" or platform.machine().upper() not in {"AMD64", "X86_64"}:
        raise ResetError("native Windows x64 required")
    if execute and not ctypes.windll.shell32.IsUserAnAdmin():
        raise ResetError("an elevated Administrator token is required for --execute")


def qualify_artifact(msi: Path, manifest_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    if not msi.is_file() or not manifest_path.is_file():
        raise ResetError("old MSI and installer manifest must be files")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ResetError("old installer manifest is unreadable") from exc
    installed = manifest.get("installed_files") if isinstance(manifest, dict) else None
    msi_record = manifest.get("msi") if isinstance(manifest, dict) else None
    if (
        manifest.get("schema_version") != 1
        or manifest.get("architecture") != CONTRACT.architecture
        or not isinstance(msi_record, dict)
        or msi_record.get("file") != msi.name
        or not isinstance(msi_record.get("sha256"), str)
        or not isinstance(installed, dict)
        or not installed
        or not all(
            isinstance(name, str)
            and name
            and not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and isinstance(digest, str)
            and len(digest) == 64
            for name, digest in installed.items()
        )
        or not isinstance(manifest.get("installed_payload_sha256"), str)
    ):
        raise ResetError("old installer manifest contract differs")
    required = {
        "CryptoHunterBackend.exe",
        "CryptoHunterFreshnessVerifier.exe",
        "CryptoHunterProvision.exe",
        "PostgreSQL/CryptoHunterPostgreSQL.exe",
    }
    if not required <= set(installed):
        raise ResetError("manifest is not the canonical CryptoHunter Windows MSI")
    if _sha256(msi) != msi_record["sha256"]:
        raise ResetError("old MSI SHA-256 differs from manifest")
    if installed_payload_digest(installed) != manifest["installed_payload_sha256"]:
        raise ResetError("installed payload digest differs from manifest")
    return manifest, installed


def qualify_installed_payload(root: Path, installed: dict[str, str]) -> None:
    if not root.is_dir() or _is_reparse_point(root):
        raise ResetError("canonical Program Files payload is absent or a reparse point")
    observed: dict[str, str] = {}
    for item in root.rglob("*"):
        if _is_reparse_point(item):
            raise ResetError("installed payload contains a reparse point")
        if item.is_file():
            observed[item.relative_to(root).as_posix()] = _sha256(item)
    if observed != installed:
        raise ResetError("installed file set or SHA-256 differs")


def _inside(root: Path, path: Path) -> bool:
    resolved_root, resolved = root.resolve(), path.resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


def _reject_windows_stream_syntax(path: Path) -> None:
    """Reject NTFS stream syntax while allowing an ordinary drive prefix."""
    windows_path = PureWindowsPath(path)
    drive = windows_path.drive
    if ":" in drive and not (len(drive) == 2 and drive[0].isalpha() and drive[1] == ":"):
        raise ResetError("receipt path must not contain Windows stream syntax")
    for component in windows_path.parts:
        if component != windows_path.anchor and ":" in component:
            raise ResetError("receipt path must not contain Windows stream syntax")


def qualify_output_path(receipt: Path) -> tuple[Path, Path]:
    """Resolve output paths and prove neither can affect product-owned trees."""
    _reject_windows_stream_syntax(receipt)
    resolved_receipt = receipt.resolve()
    uninstall_log = resolved_receipt.with_suffix(".uninstall.log")
    forbidden_roots = tuple(path.resolve() for path in canonical_paths())
    for output in (resolved_receipt, uninstall_log):
        if any(
            _inside(root, candidate)
            for root in forbidden_roots
            for candidate in (output, output.parent)
        ):
            raise ResetError("receipt and uninstall log must be outside product-owned trees")
    return resolved_receipt, uninstall_log


def qualify_preserved_program_data(program_files: Path, machine: Path) -> dict[str, Any]:
    if not machine.is_dir() or _is_reparse_point(machine):
        raise ResetError("canonical ProgramData base is absent or a reparse point")
    try:
        record = json.loads(
            (machine / ".stage9-install-ownership.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ResetError("committed ownership record absent or malformed") from exc
    expected = {
        "schema_version": 1,
        "state": "COMMITTED",
        "program_files": str(program_files.resolve()),
        "program_data": str(machine.resolve()),
        "service_names": list(SERVICES),
    }
    if not isinstance(record, dict) or any(record.get(k) != v for k, v in expected.items()):
        raise ResetError("committed ownership record differs")
    resources = record.get("resources_created")
    if not isinstance(resources, list) or not all(isinstance(value, str) for value in resources):
        raise ResetError("ownership resources_created is malformed")
    for value in resources:
        resource = Path(value)
        if not resource.is_absolute() or not _inside(machine, resource):
            raise ResetError("ownership resource escapes canonical ProgramData")
    if {item.name for item in machine.iterdir()} != BASE_TOP_LEVEL:
        raise ResetError("ProgramData is not the exact base-install shape")
    for item in (machine, *machine.rglob("*")):
        if _is_reparse_point(item) or not _inside(machine, item):
            raise ResetError("ProgramData contains a reparse point or path escape")
    for marker in AUTHORITY_MARKERS:
        if (machine / marker).exists():
            raise ResetError(f"production authority/enrollment marker exists: {marker}")
    if transaction_path(machine).exists():
        raise ResetError("Stage-9 transaction journal exists")
    if safe_failure_path().exists() or (machine.parent / STARTUP_FAILURE_DIAGNOSTIC).exists():
        raise ResetError("unresolved install failure diagnostic exists")
    return record


def query_services() -> dict[str, dict[str, str]]:
    names = ",".join(f"'{name}'" for name in SERVICES)
    script = (
        f"@(Get-CimInstance Win32_Service | Where-Object {{$_.Name -in @({names})}} | "
        "Select-Object Name,PathName,StartMode,State,StartName,Dependencies) | "
        "ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    values = json.loads(result.stdout or "[]")
    if isinstance(values, dict):
        values = [values]
    return {value["Name"]: value for value in values}


def qualify_services(program_files: Path, services: dict[str, dict[str, Any]]) -> None:
    expected = {
        CONTRACT.postgresql_service: (
            program_files / "PostgreSQL/CryptoHunterPostgreSQL.exe",
            "Auto",
            CONTRACT.postgresql_identity,
            None,
        ),
        CONTRACT.backend_service: (
            program_files / "CryptoHunterBackend.exe",
            "Manual",
            CONTRACT.backend_identity,
            CONTRACT.postgresql_service,
        ),
        CONTRACT.verifier_service: (
            program_files / "CryptoHunterFreshnessVerifier.exe",
            "Manual",
            CONTRACT.verifier_identity,
            None,
        ),
    }
    if set(services) != set(expected):
        raise ResetError("SCM base service set differs")
    for name, (binary, start, account, dependency) in expected.items():
        value = services[name]
        path = str(value.get("PathName", "")).strip('"')
        dependencies = value.get("Dependencies") or []
        if isinstance(dependencies, str):
            dependencies = [dependencies]
        if (
            path.casefold() != str(binary).casefold()
            or value.get("StartMode") != start
            or str(value.get("StartName", "")).casefold() != account.casefold()
            or (dependency is not None and dependencies != [dependency])
            or (dependency is None and dependencies)
            or (name != CONTRACT.postgresql_service and value.get("State") != "Stopped")
        ):
            raise ResetError(f"SCM contract differs: {name}")


def _product_code(msi: Path) -> str:
    escaped_msi = str(msi).replace("'", "''")
    script = (
        "$i=New-Object -ComObject WindowsInstaller.Installer;"
        f"$d=$i.GetType().InvokeMember('OpenDatabase','InvokeMethod',$null,$i,@('{escaped_msi}',0));"
        "$v=$d.OpenView(\"SELECT `Value` FROM `Property` WHERE `Property`='ProductCode'\");"
        "$v.Execute();$v.Fetch().StringData(1)"
    )
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def require_product_unregistered(product_code: str) -> None:
    state = ctypes.windll.msi.MsiQueryProductStateW(product_code)
    if state != -1:
        raise ResetError(f"Windows Installer product remains registered (state {state})")


def uninstall(msi: Path, log: Path) -> int:
    result = subprocess.run(
        ["msiexec.exe", "/x", str(msi), "/qn", "/norestart", "/l*v", str(log)],
        timeout=600,
    )
    return result.returncode


def guarded_purge(machine: Path) -> None:
    """Delete only the canonical tree after its final full qualification."""
    _, canonical = canonical_paths()
    if machine.resolve() != canonical.resolve():
        raise ResetError("purge target is not canonical ProgramData")
    shutil.rmtree(machine)


def _receipt_template(
    msi: Path, manifest_path: Path, manifest: dict[str, Any], execute: bool
) -> dict[str, Any]:
    program_files, program_data = canonical_paths()
    return {
        "schema_version": 1,
        "operation": OPERATION,
        "executed": execute,
        "old_msi_sha256": _sha256(msi),
        "old_manifest_sha256": _sha256(manifest_path),
        "old_installed_payload_sha256": manifest["installed_payload_sha256"],
        "program_files": str(program_files),
        "program_data": str(program_data),
        "preflight": "NOT_RUN",
        "uninstall_exit_code": "NOT_RUN",
        "post_uninstall": "NOT_RUN",
        "authority_absent_recheck": "NOT_RUN",
        "program_data_purge": "NOT_RUN",
        "final_clean_preconditions": "NOT_RUN",
        "preserved_program_data_remains": "YES",
        "result": "FAIL",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    receipt_path, log = qualify_output_path(args.receipt)
    args._qualified_receipt = receipt_path
    require_host(execute=args.execute)
    msi, manifest_path = args.installed_msi.resolve(), args.installed_manifest.resolve()
    manifest, installed = qualify_artifact(msi, manifest_path)
    receipt = _receipt_template(msi, manifest_path, manifest, args.execute)
    program_files, machine = canonical_paths()
    try:
        qualify_installed_payload(program_files, installed)
        qualify_preserved_program_data(program_files, machine)
        services = query_services()
        qualify_services(program_files, services)
        receipt["preflight"] = "PASS"
        for line in (
            "PRE_ENROLLMENT_BASE_IDENTITY = PASS",
            "INSTALLED_PAYLOAD_MATCH = PASS",
            "COMMITTED_OWNERSHIP = PASS",
            "AUTHORITY_ABSENT = PASS",
            "PROGRAMDATA_SHAPE = PASS",
            "SCM_BASE_STATE = PASS",
            "RESET_ELIGIBLE = YES",
        ):
            print(line)
        if not args.execute:
            receipt["result"] = "PASS"
            print("MUTATION = NOT_PERFORMED")
            return receipt
        log.parent.mkdir(parents=True, exist_ok=True)
        product_code = _product_code(msi)
        code = uninstall(msi, log)
        receipt["uninstall_exit_code"] = code
        if code != 0:
            raise ResetError(f"normal MSI uninstall failed with exit code {code}")
        remaining = query_services()
        if program_files.exists() or remaining:
            raise ResetError("post-uninstall MSI-owned resources remain")
        require_product_unregistered(product_code)
        receipt["post_uninstall"] = "PASS"
        qualify_preserved_program_data(program_files, machine)
        receipt["authority_absent_recheck"] = "PASS"
        guarded_purge(machine)
        receipt["program_data_purge"] = "PASS"
        receipt["preserved_program_data_remains"] = "NO"
        if (
            program_files.exists()
            or machine.exists()
            or transaction_path(machine).exists()
            or query_services()
        ):
            raise ResetError("final clean-target proof failed")
        receipt["final_clean_preconditions"] = "PASS"
        receipt["result"] = "PASS"
        return receipt
    except Exception:
        receipt["preserved_program_data_remains"] = "YES" if machine.exists() else "NO"
        raise
    finally:
        args._receipt_value = receipt


def _write_receipt(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-msi", required=True, type=Path)
    parser.add_argument("--installed-manifest", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    args._receipt_value = None
    args._qualified_receipt = None
    try:
        value = run(args)
    except Exception as exc:
        value = args._receipt_value
        if value is not None:
            value["result"] = "FAIL"
            _write_receipt(args._qualified_receipt, value)
        print(f"PRE_ENROLLMENT_RESET = FAIL ({type(exc).__name__})", file=sys.stderr)
        return 1
    _write_receipt(args._qualified_receipt, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
