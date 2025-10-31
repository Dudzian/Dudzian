"""Utilities for building OEM distribution bundles."""

from .build_core_bundle import (  # noqa: F401
    BundleInputs,
    CoreBundleBuilder,
    SignatureManager,
    build_from_cli as build_core_bundle_from_cli,
)
from .build_strategy_bundle import (  # noqa: F401
    StrategyBundleBuilder,
    build_from_cli as build_strategy_bundle_from_cli,
)
from .build_pyinstaller_bundle import build_bundle as build_pyinstaller_bundle  # noqa: F401
from .desktop_installer import (  # noqa: F401
    DesktopInstallerBuilder,
    build_from_cli as build_desktop_installer_from_cli,
)

# Backwards compatibility: expose ``build_from_cli`` used by legacy tests.
build_from_cli = build_core_bundle_from_cli

__all__ = [
    "BundleInputs",
    "CoreBundleBuilder",
    "SignatureManager",
    "StrategyBundleBuilder",
    "build_pyinstaller_bundle",
    "build_core_bundle_from_cli",
    "build_strategy_bundle_from_cli",
    "DesktopInstallerBuilder",
    "build_desktop_installer_from_cli",
    "build_from_cli",
]
