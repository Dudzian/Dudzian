"""Frozen macOS boundary; launchd implementation is intentionally not fabricated."""

SERVICE_MANAGER = "launchd"


class MacOSDeploymentNotImplemented(NotImplementedError):
    pass


def qualify_live() -> None:
    raise MacOSDeploymentNotImplemented("launchd, permissions, IPC and local auth pending")

