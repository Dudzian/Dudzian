"""Generate a safe Windows desktop build environment report for GymOS."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "build" / "reports" / "windows_environment_report.json"
APP_NAME = "GymOS"
DESKTOP_ENTRYPOINT = "ui.pyside_app"
QML_ASSETS = ("ui/pyside_app/qml", "ui/qml")
QT_PLUGINS = ("platforms", "imageformats", "iconengines", "styles", "qml")
DEPENDENCY_FILES = (
    "requirements.txt",
    "pyproject.toml",
    "deploy/packaging/requirements-desktop.txt",
    "deploy/packaging/requirements-desktop.lock",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_version() -> str:
    return _package_version("dudzian-bot") or "0+unknown"


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_sha() -> str | None:
    env_sha = os.environ.get("GITHUB_SHA")
    if env_sha:
        return env_sha
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _list_files(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return [p.relative_to(ROOT).as_posix() for p in sorted(directory.rglob("*")) if p.is_file()]


def build_report() -> dict[str, Any]:
    dependency_files: list[dict[str, str]] = []
    for relative in DEPENDENCY_FILES:
        path = ROOT / relative
        if path.exists() and path.is_file():
            dependency_files.append({"path": relative, "sha256": _sha256(path)})

    return {
        "system": platform.system(),
        "architecture": platform.machine(),
        "windows_version": platform.version() if platform.system() == "Windows" else None,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pyside6_version": _package_version("PySide6"),
        "packager": "PyInstaller",
        "packager_version": _package_version("pyinstaller"),
        "desktop_entrypoint": DESKTOP_ENTRYPOINT,
        "source_run_command": "python -m ui.pyside_app",
        "safe_smoke_command": "python -m ui.pyside_app --smoke-test --offscreen",
        "qml_assets": {asset: _list_files(ROOT / asset) for asset in QML_ASSETS},
        "required_qt_plugins": list(QT_PLUGINS),
        "dependency_and_lock_files": dependency_files,
        "output_name": f"{APP_NAME}.exe",
        "application_name": APP_NAME,
        "application_version": _project_version(),
        "commit_sha": _git_sha(),
        "notes": [
            "offscreen smoke is an automated CI safety check and does not replace manual GUI testing",
            "report intentionally excludes tokens, credentials, secrets, local databases, and user state",
        ],
    }


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(REPORT_PATH.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
