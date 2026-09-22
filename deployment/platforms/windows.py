"""Windows reference boundary. Privileged provisioning remains installer-owned."""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath

from .contracts import DeploymentPaths

SERVICE_NAME = "CryptoHunterBackend"
# Virtual service account: distinct from an interactive user and not LocalSystem.
SERVICE_IDENTITY = rf"NT SERVICE\{SERVICE_NAME}"
IPC_NAME = rf"\\.\pipe\{SERVICE_NAME}\freshness-v1"


class WindowsDeploymentNotQualified(RuntimeError):
    """A Windows-only proof is unavailable or failed closed."""


def resolve_paths(environment: dict[str, str] | None = None) -> DeploymentPaths:
    """Resolve Windows-native roots; fail rather than inventing drive paths."""
    if os.name != "nt":
        raise WindowsDeploymentNotQualified("native Windows paths require a real Windows host")
    values = os.environ if environment is None else environment
    required = ("ProgramFiles", "ProgramData", "LOCALAPPDATA")
    missing = [name for name in required if not values.get(name)]
    if missing:
        raise WindowsDeploymentNotQualified(f"missing Windows known-folder environment: {missing}")
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


def qualify_acl() -> None:
    """Fail closed until native ACL inspection is implemented and run on Windows."""
    raise WindowsDeploymentNotQualified(
        "native DACL qualification is not implemented; POSIX modes are not a substitute"
    )


def qualify_local_principal_authentication() -> None:
    """Fail closed: localhost/password or caller claims are deliberately unacceptable."""
    raise WindowsDeploymentNotQualified(
        "reviewed Windows PostgreSQL authenticated-principal mechanism is not implemented"
    )
