"""Canonical Stage-9 Windows production-installer contract.

The desktop ZIP packager is intentionally separate.  This package owns the
per-machine x64 MSI and never performs installation-time downloads.
"""

from .contract import CONTRACT, InstallerContract

__all__ = ["CONTRACT", "InstallerContract"]
