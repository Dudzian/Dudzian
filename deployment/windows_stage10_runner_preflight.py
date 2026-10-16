"""Read-only preparation checks; this report can never qualify Stage 10.

Only ``--live`` collects host/API facts. A replayed snapshot is rehearsal input,
not a claim that the approved physical Windows runner was reached. No command in
this module installs software, changes services, reboots, or accesses TPM/keys.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess
import stat
import sys
from typing import Any, cast
import uuid

import yaml

_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
REQUIRED_LABELS = ("self-hosted", "Windows", "X64", "cryptohunter-stage10")
# WriteData, AppendData, WriteExtendedAttributes, WriteAttributes, Delete,
# ChangePermissions, TakeOwnership. Read-only grants are deliberately allowed.
_WRITE_RIGHTS = 2 | 4 | 16 | 256 | 65536 | 262144 | 524288
_REQUIRED_CONFIG = {
    "schema_version",
    "repository",
    "hostname",
    "machine_guid",
    "hardware_uuid",
    "runner_id",
    "runner_name",
    "runner_service_name",
    "runner_root",
    "runner_service_account",
    "runner_group_id",
    "trusted_workflows",
}


class RunnerPreflightError(ValueError):
    """An input or read-only observation cannot be trusted."""


def _windows_path(value: Any) -> bool:
    if not isinstance(value, str) or re.match(r"^[A-Za-z]:\\", value) is None:
        return False
    path = PureWindowsPath(value)
    return path.is_absolute() and all(part not in {".", ".."} for part in value.split("\\"))


def validate_runner_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict) or set(config) != _REQUIRED_CONFIG:
        raise RunnerPreflightError(
            "configuration must contain exactly the documented approval fields"
        )
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise RunnerPreflightError("unsupported configuration schema_version")
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", str(config["repository"])) is None:
        raise RunnerPreflightError("repository must be owner/repository")
    for key in ("hostname", "runner_name", "runner_service_name", "runner_service_account"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise RunnerPreflightError(f"missing approved {key}")
    for key in ("hostname", "runner_name", "runner_service_name"):
        if re.fullmatch(r"[A-Za-z0-9_.-]+", config[key]) is None:
            raise RunnerPreflightError(f"invalid approved {key}")
    for key in ("runner_id", "runner_group_id"):
        if type(config[key]) is not int or config[key] <= 0:
            raise RunnerPreflightError(f"{key} must be a positive integer")
    for key in ("machine_guid", "hardware_uuid"):
        try:
            parsed = uuid.UUID(config[key])
        except (TypeError, ValueError, AttributeError) as exc:
            raise RunnerPreflightError(f"invalid approved {key}") from exc
        if parsed.int == 0:
            raise RunnerPreflightError(f"invalid approved {key}")
    if not _windows_path(config["runner_root"]):
        raise RunnerPreflightError(
            "runner_root must be an absolute local Windows path without traversal"
        )
    workflows = config["trusted_workflows"]
    prefix = re.escape(config["repository"]) + r"/\.github/workflows/"
    if (
        not isinstance(workflows, list)
        or not workflows
        or len(set(str(item) for item in workflows)) != len(workflows)
        or any(
            not isinstance(item, str)
            or re.fullmatch(prefix + r"[A-Za-z0-9_-]+\.ya?ml@refs/heads/main", item) is None
            for item in workflows
        )
    ):
        raise RunnerPreflightError(
            "trusted_workflows must name only approved repository workflows at main"
        )
    return dict(config)


# Values enter the script through the process environment, never PowerShell code.
# The .runner file is filtered to public registration fields before returning it.
_HOST_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$root = $env:CRYPTOHUNTER_PREFLIGHT_ROOT
$serviceName = $env:CRYPTOHUNTER_PREFLIGHT_SERVICE
$os = Get-CimInstance Win32_OperatingSystem
$computer = Get-CimInstance Win32_ComputerSystem
$hardware = Get-CimInstance Win32_ComputerSystemProduct
$service = Get-CimInstance Win32_Service | Where-Object Name -CEQ $serviceName
if (@($service).Count -ne 1) { throw 'exactly one approved runner SCM service is required' }
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$($service.ProcessId)"
$registration = Get-Content -Raw -LiteralPath (Join-Path $root '.runner') | ConvertFrom-Json
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
$serviceSid = ([Security.Principal.NTAccount]::new($service.StartName)).Translate(
  [Security.Principal.SecurityIdentifier]).Value
$acl = @((Get-Acl -LiteralPath $root).Access | ForEach-Object {
  @{ sid=$_.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value;
     rights=[int]$_.FileSystemRights; access_type=$_.AccessControlType.ToString() }
})
@{
  os='Windows'; architecture=$os.OSArchitecture; hostname=$computer.Name;
  native_processor_architectures=@(Get-CimInstance Win32_Processor | ForEach-Object { [int]$_.Architecture });
  is_64_bit_os=[Environment]::Is64BitOperatingSystem;
  is_64_bit_process=[Environment]::Is64BitProcess; process_architecture=$env:PROCESSOR_ARCHITECTURE;
  machine_guid=(Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Cryptography').MachineGuid;
  hardware_uuid=$hardware.UUID; manufacturer=$computer.Manufacturer; model=$computer.Model;
  runner_registration=@{ id=$registration.agentId; name=$registration.agentName;
    github_url=$registration.gitHubUrl };
  service=@{ name=$service.Name; start_mode=$service.StartMode; state=$service.State;
    process_id=$service.ProcessId; path=$service.PathName; account=$service.StartName;
    account_sid=$serviceSid; process_path=$process.ExecutablePath };
  identity=@{ sid=$identity.User.Value; groups=@($identity.Groups | ForEach-Object Value);
    elevated=$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator) };
  runner_root_acl=$acl
} | ConvertTo-Json -Depth 8 -Compress
"""


def _read_json_command(command: list[str], *, env: dict[str, str] | None = None) -> Any:
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=45, env=env
    )
    return json.loads(result.stdout.lstrip("\ufeff"))


def _github_get(endpoint: str) -> Any:
    # gh's configured authentication is used only for GET; no token is printed.
    return _read_json_command(
        ["gh", "api", "--method", "GET", "--hostname", "github.com", endpoint]
    )


def _collect_host(config: dict[str, Any]) -> dict[str, Any]:
    if os.name != "nt":
        raise RunnerPreflightError("actual Windows runner is unavailable on this host")
    env = dict(os.environ)
    env["CRYPTOHUNTER_PREFLIGHT_ROOT"] = config["runner_root"]
    env["CRYPTOHUNTER_PREFLIGHT_SERVICE"] = config["runner_service_name"]
    encoded = base64.b64encode(_HOST_SCRIPT.encode("utf-16-le")).decode("ascii")
    result = _read_json_command(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded], env=env
    )
    if not isinstance(result, dict):
        raise RunnerPreflightError("host collector did not return a JSON object")
    return result


def _collect_github(config: dict[str, Any]) -> dict[str, Any]:
    owner = config["repository"].split("/")[0]
    group_endpoint = f"orgs/{owner}/actions/runner-groups/{config['runner_group_id']}"
    github = {
        "runner": _github_get(f"orgs/{owner}/actions/runners/{config['runner_id']}"),
        "group": _github_get(group_endpoint),
        # --paginate is deliberately not needed: total_count must equal the
        # received complete list below; incomplete responses cannot pass.
        "group_runners": _github_get(group_endpoint + "/runners?per_page=100"),
        "group_repositories": _github_get(group_endpoint + "/repositories?per_page=100"),
        "workflows": {},
    }
    for workflow in config["trusted_workflows"]:
        path = workflow.split("/.github/", 1)[1].split("@", 1)[0]
        response = _github_get(f"repos/{config['repository']}/contents/.github/{path}?ref=main")
        if not isinstance(response, dict) or response.get("encoding") != "base64":
            raise RunnerPreflightError("GitHub did not return workflow source content")
        github["workflows"][workflow] = base64.b64decode(
            response["content"], validate=False
        ).decode("utf-8")
    return github


def _complete_list(data: Any, key: str) -> list[Any] | None:
    if not isinstance(data, dict) or not isinstance(data.get(key), list):
        return None
    items = cast(list[Any], data[key])
    if type(data.get("total_count")) is not int or data["total_count"] != len(items):
        return None
    return items


def _manual_workflow(source: Any) -> bool:
    if not isinstance(source, str):
        return False
    try:
        data = yaml.safe_load(source)
    except yaml.YAMLError:
        return False
    if not isinstance(data, dict):
        return False
    # PyYAML's YAML 1.1 resolver treats the GitHub Actions key `on` as True.
    triggers = data.get("on", data.get(True))
    return triggers == "workflow_dispatch" or (
        isinstance(triggers, dict) and set(triggers) == {"workflow_dispatch"}
    )


def run_runner_preflight(
    config: Any,
    *,
    live: bool = False,
    snapshot: Any = None,
) -> dict[str, Any]:
    """Collect live reads or rehearse a snapshot; failures always remain visible.

    A snapshot cannot select live verification by including booleans/metadata.
    GitHub registration, group access policy and workflow sources are collected
    independently of the locally supplied operator approval configuration.
    """
    report: dict[str, Any] = {
        "schema_version": 1,
        "report_type": "windows-stage10-read-only-runner-preflight",
        "mode": "live" if live else "rehearsal",
        "stage10_qualification": "NOT_RUN",
        "stage10_execution_authorized": False,
        "checks": [],
    }

    def check(name: str, status: str, details: str, *, observed: bool = True) -> None:
        if status == "PASS" and observed and not live:
            status = "UNVERIFIED"
            details = "Snapshot is consistent only; actual runner was not reached. " + details
        report["checks"].append({"name": name, "status": status, "details": details})

    def finish() -> dict[str, Any]:
        statuses = {item["status"] for item in report["checks"]}
        report["overall_status"] = next(
            (status for status in ("FAIL", "BLOCKED", "UNVERIFIED") if status in statuses), "PASS"
        )
        return report

    try:
        approved = validate_runner_config(config)
    except RunnerPreflightError as exc:
        check("approved_configuration", "BLOCKED", str(exc))
        return finish()
    check(
        "approved_configuration", "PASS", "Approval configuration schema is valid.", observed=False
    )
    if live and snapshot is not None:
        check("live_collection", "FAIL", "Live verification cannot consume a saved snapshot.")
        return finish()
    observations = snapshot if isinstance(snapshot, dict) else {}
    if live:
        observations = {}
        for name, collector in (("host", _collect_host), ("github", _collect_github)):
            try:
                observations[name] = collector(approved)
            except (
                RunnerPreflightError,
                OSError,
                ValueError,
                TypeError,
                KeyError,
                subprocess.SubprocessError,
            ) as exc:
                # Avoid subprocess stderr: credentials or private API data can
                # appear there. The exception type and failed read are enough.
                check(
                    name + "_access",
                    "BLOCKED",
                    f"Read-only {name} collection failed ({type(exc).__name__}).",
                )
    host = observations.get("host")
    github = observations.get("github")
    if not isinstance(host, dict):
        check(
            "actual_runner",
            "BLOCKED",
            "No actual approved Windows host was observed; provisioning/access remains required.",
        )
    else:
        virtual = re.search(
            r"virtual|vmware|vbox|kvm|qemu|xen|parallels|amazon ec2|google compute|hvm domU",
            f"{host.get('manufacturer', '')} {host.get('model', '')}",
            re.IGNORECASE,
        )
        hardware = all(
            isinstance(host.get(key), str) and host[key].strip()
            for key in ("manufacturer", "model")
        )
        identity_ok = (
            host.get("hostname") == approved["hostname"]
            and str(host.get("machine_guid", "")).lower() == approved["machine_guid"].lower()
            and str(host.get("hardware_uuid", "")).lower() == approved["hardware_uuid"].lower()
        )
        processors = host.get("native_processor_architectures")
        x64 = (
            host.get("os") == "Windows"
            and host.get("is_64_bit_os") is True
            and host.get("is_64_bit_process") is True
            and host.get("process_architecture") == "AMD64"
            and isinstance(processors, list)
            and bool(processors)
            and all(type(item) is int and item == 9 for item in processors)
        )
        check(
            "windows_x64",
            "PASS" if x64 else "FAIL",
            "Live OS and collector must be native Windows x64: Win32_Processor Architecture=9, 64-bit OS/process, AMD64 process. ARM64/emulation/WOW64 and labels alone cannot pass.",
        )
        check(
            "approved_physical_host",
            "PASS" if identity_ok and hardware and not virtual else "FAIL",
            "Hostname, MachineGuid and hardware UUID must match the approved inventory; missing/virtual hardware identifiers fail. Inventory matching is not hardware attestation.",
        )
        registration = host.get("runner_registration", {})
        owner = approved["repository"].split("/")[0]
        registration_ok = isinstance(registration, dict) and (
            type(registration.get("id")) is int
            and registration["id"] == approved["runner_id"]
            and registration.get("name") == approved["runner_name"]
            and registration.get("github_url") == f"https://github.com/{owner}"
        )
        check(
            "local_runner_identity",
            "PASS" if registration_ok else "FAIL",
            "Public .runner identity must match the approved organization registration; credentials are not read.",
        )
        service = host.get("service", {})
        if not isinstance(service, dict):
            service = {}
        executable = str(PureWindowsPath(approved["runner_root"]) / "bin" / "RunnerService.exe")
        service_ok = (
            service.get("name") == approved["runner_service_name"]
            and service.get("start_mode") == "Auto"
            and service.get("state") == "Running"
            and type(service.get("process_id")) is int
            and service["process_id"] > 0
            and (
                service.get("path") == f'"{executable}"'
                or (
                    not any(char.isspace() for char in executable)
                    and service.get("path") == executable
                )
            )
            and service.get("process_path") == executable
            and service.get("account") == approved["runner_service_account"]
        )
        check(
            "automatic_runner_service",
            "PASS" if service_ok else "FAIL",
            "SCM must show the approved running automatic runner service/account and exact RunnerService.exe path. Reboot persistence is configured, not reboot-tested.",
        )
        identity = host.get("identity", {})
        if not isinstance(identity, dict):
            identity = {}
        groups = identity.get("groups")
        privilege_ok = (
            isinstance(service.get("account_sid"), str)
            and service["account_sid"].startswith("S-1-5-21-")
            and identity.get("sid") == service["account_sid"]
            and identity.get("elevated") is False
            and isinstance(groups, list)
            and "S-1-5-32-544" not in groups
        )
        check(
            "least_privilege_runner",
            "PASS" if privilege_ok else "FAIL",
            "Collect as the approved service identity: it must be a non-administrator user with a running SCM service. Future elevated MSI installation requires a separate operator approval.",
        )
        trusted_write_sids = {"S-1-5-18", "S-1-5-32-544"}
        if isinstance(service.get("account_sid"), str):
            trusted_write_sids.add(service["account_sid"])
        acl = host.get("runner_root_acl")
        acl_ok = (
            isinstance(acl, list)
            and bool(acl)
            and all(
                isinstance(entry, dict)
                and isinstance(entry.get("sid"), str)
                and type(entry.get("rights")) is int
                and isinstance(entry.get("access_type"), str)
                and entry["access_type"] in {"Allow", "Deny"}
                and not (
                    entry["access_type"] == "Allow"
                    and entry["sid"] not in trusted_write_sids
                    and entry["rights"] & _WRITE_RIGHTS
                )
                for entry in acl
            )
        )
        check(
            "runner_directory_acl",
            "PASS" if acl_ok else "FAIL",
            "Runner directory ACL must be readable; only SYSTEM, Administrators and the approved runner service identity may have write grants (conservative Allow-rule check).",
        )
    if not isinstance(github, dict):
        check(
            "github_registration_and_isolation",
            "UNVERIFIED",
            "Independent GitHub registration/group/workflow reads are unavailable; supplied approval booleans cannot establish labels or PR isolation.",
        )
    else:
        runner = github.get("runner", {})
        if not isinstance(runner, dict):
            runner = {}
        labels = runner.get("labels")
        labels_ok = (
            isinstance(labels, list)
            and len(labels) == len(REQUIRED_LABELS)
            and all(isinstance(item, dict) for item in labels)
            and sorted(str(item.get("name")) for item in labels if isinstance(item, dict))
            == sorted(REQUIRED_LABELS)
        )
        runner_ok = (
            type(runner.get("id")) is int
            and runner["id"] == approved["runner_id"]
            and runner.get("name") == approved["runner_name"]
            and runner.get("os") == "windows"
            and runner.get("status") == "online"
            and labels_ok
        )
        check(
            "github_runner_registration",
            "PASS" if runner_ok else "FAIL",
            "Independent GitHub registration must be online with the approved id/name and exactly self-hosted, Windows, X64, cryptohunter-stage10 labels.",
        )
        group = github.get("group", {})
        members = _complete_list(github.get("group_runners"), "runners")
        repositories = _complete_list(github.get("group_repositories"), "repositories")
        sources = github.get("workflows", {})
        isolation_ok = (
            isinstance(group, dict)
            and group.get("id") == approved["runner_group_id"]
            and group.get("visibility") == "selected"
            and group.get("restricted_to_workflows") is True
            and isinstance(group.get("selected_workflows"), list)
            and all(isinstance(item, str) for item in group["selected_workflows"])
            and sorted(group["selected_workflows"]) == sorted(approved["trusted_workflows"])
            and members is not None
            and len(members) == 1
            and isinstance(members[0], dict)
            and members[0].get("id") == approved["runner_id"]
            and repositories is not None
            and len(repositories) == 1
            and isinstance(repositories[0], dict)
            and repositories[0].get("full_name") == approved["repository"]
            and isinstance(sources, dict)
            and all(_manual_workflow(sources.get(item)) for item in approved["trusted_workflows"])
        )
        check(
            "untrusted_pr_isolation",
            "PASS" if isolation_ok else "FAIL",
            "A dedicated GitHub runner group must restrict the sole approved runner/repository to the exact main-only workflow allowlist; each live workflow must expose only workflow_dispatch. PR-triggered workflows or incomplete API lists cannot prove isolation.",
        )
    return finish()


def _write_new_report(path: Path, report: dict[str, Any], config: Any) -> None:
    """Create an operator-selected report without replacing any existing input/file."""
    if path.suffix.lower() != ".json" or ".." in path.parts:
        raise RunnerPreflightError("report must be a new JSON file without path traversal")
    forbidden = []
    if isinstance(config, dict) and isinstance(config.get("runner_root"), str):
        forbidden.append(PureWindowsPath(config["runner_root"]))
    for variable in ("ProgramFiles", "ProgramFiles(x86)", "ProgramData"):
        root = os.environ.get(variable)
        if root:
            forbidden.append(PureWindowsPath(root) / "CryptoHunter")
    absolute = path.absolute()
    if any(
        PureWindowsPath(str(candidate)).is_relative_to(root)
        for candidate in (path, absolute)
        for root in forbidden
    ):
        raise RunnerPreflightError("report cannot be written inside runner or product directories")
    for ancestor in (absolute, *absolute.parents):
        try:
            metadata = ancestor.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or (
            getattr(metadata, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise RunnerPreflightError("report paths cannot contain symlinks or reparse points")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = None
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        snapshot = json.loads(args.snapshot.read_text(encoding="utf-8")) if args.snapshot else None
        report = run_runner_preflight(config, live=args.live, snapshot=snapshot)
    except (OSError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": 1,
            "report_type": "windows-stage10-read-only-runner-preflight",
            "overall_status": "BLOCKED",
            "stage10_qualification": "NOT_RUN",
            "stage10_execution_authorized": False,
            "checks": [{"name": "input", "status": "BLOCKED", "details": str(exc)}],
        }
    try:
        _write_new_report(args.output, report, config)
    except (OSError, RunnerPreflightError) as exc:
        print(f"BLOCKED: cannot create a safe new report: {exc}", file=sys.stderr)
        return 1
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
