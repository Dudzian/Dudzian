"""Authoritative Windows desktop build naming for the CryptoHunter app."""

from __future__ import annotations

from pathlib import Path

PYTHON_PACKAGE_NAME = "dudzian-bot"
PRODUCT_NAME = "CryptoHunter"
EXE_NAME = f"{PRODUCT_NAME}.exe"
ONEDIR_NAME = PRODUCT_NAME
SPEC_FILE_NAME = f"{PRODUCT_NAME}.spec"
WINDOWS_ARTIFACT_PREFIX = f"{PRODUCT_NAME}-windows"
WINDOWS_REPORTS_ARTIFACT_PREFIX = f"{WINDOWS_ARTIFACT_PREFIX}-reports"


def onedir_path(output_root: str | Path = "build/output") -> Path:
    """Return the expected PyInstaller onedir output path."""

    return Path(output_root) / ONEDIR_NAME
