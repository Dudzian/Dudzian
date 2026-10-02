"""Windows reference boundary. Privileged provisioning remains installer-owned."""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath

from .contracts import DeploymentPaths

SERVICE_NAME = "CryptoHunterBackend"
# Virtual service account: distinct from an interactive user and not LocalSystem.
SERVICE_IDENTITY = rf"NT SERVICE\{SERVICE_NAME}"
IPC_NAME = rf"\\.\pipe\{SERVICE_NAME}\freshness-v1"
PRODUCTION_TRUST_DIRECTORY_NAME = "ProductionTrust"


class WindowsDeploymentNotQualified(RuntimeError):
    """A Windows-only proof is unavailable or failed closed."""


def resolve_paths(environment: dict[str, str] | None = None) -> DeploymentPaths:
    """Resolve Windows-native roots; fail rather than inventing drive paths."""
    if os.name != "nt":
        raise WindowsDeploymentNotQualified(
            "native Windows paths require a real Windows host"
        )
    values = os.environ if environment is None else environment
    required = ("ProgramFiles", "ProgramData", "LOCALAPPDATA")
    missing = [name for name in required if not values.get(name)]
    if missing:
        raise WindowsDeploymentNotQualified(
            f"missing Windows known-folder environment: {missing}"
        )
    invalid = [
        name for name in required if not PureWindowsPath(values[name]).is_absolute()
    ]
    if invalid:
        raise WindowsDeploymentNotQualified(
            f"Windows known-folder paths must be fully qualified: {invalid}"
        )
    install = Path(values["ProgramFiles"]) / "CryptoHunter"
    machine = Path(values["ProgramData"]) / "CryptoHunter"
    return DeploymentPaths(
        install=install,
        state=machine / "State",
        configuration=machine / "Config",
        logs=machine / "Logs",
        runtime=machine / "Runtime",
        update_staging=machine / "Updates",
        gui_user_state=Path(values["LOCALAPPDATA"]) / "CryptoHunter",
    )


def production_trust_package_path(
    ceremony_id: str, environment: dict[str, str] | None = None
) -> Path:
    """Canonical installer-owned, runtime-read-only public trust package location."""
    return (
        resolve_paths(environment).configuration
        / PRODUCTION_TRUST_DIRECTORY_NAME
        / ceremony_id
    )


def static_path_layout(
    program_files: str, program_data: str, local_app_data: str
) -> tuple[PureWindowsPath, ...]:
    """Describe layout syntax only; this is deliberately not live filesystem evidence."""
    machine = PureWindowsPath(program_data) / "CryptoHunter"
    return (
        PureWindowsPath(program_files) / "CryptoHunter",
        machine / "State",
        machine / "Config",
        machine / "Logs",
        machine / "Runtime",
        machine / "Updates",
        PureWindowsPath(local_app_data) / "CryptoHunter",
    )


def qualify_acl(service_sid: str | None = None) -> dict[str, str]:
    """Invoke only the read-only native Windows DACL boundary."""
    from deployment.windows_dacl_qualification import (
        qualify_acl as read_only_qualify_acl,
    )

    return read_only_qualify_acl(service_sid)


def qualify_local_principal_authentication(
    scratch_parent: Path | None = None,
) -> dict[str, str]:
    """Delegate to the reviewed live mTLS qualification; never repair production state."""
    if scratch_parent is None:
        raise WindowsDeploymentNotQualified(
            "a caller-owned Stage-8 scratch parent and live Windows proof are required"
        )
    from deployment.windows_stage8_postgresql_probe import run_probe

    return run_probe(scratch_parent)
