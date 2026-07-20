"""Validate a built GymOS Windows onedir artifact by path names only."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

FORBIDDEN_PARTS = (
    ".env",
    "credentials",
    "credential",
    "secrets",
    "secret",
    "tokens",
    "token",
    ".git",
    ".venv",
)
FORBIDDEN_SUFFIXES = (".db", ".sqlite", ".sqlite3")
FORBIDDEN_PROJECT_DIRS = {"logs", "log", "cache", ".cache", "__pycache__"}
REQUIRED_ROOT_FILES = ("GymOS.exe", "BUILD_INFO.txt")
REQUIRED_DATA_FILES = (
    "ui/config/preview_local.yaml",
    "ui/pyside_app/qml/MainWindow.qml",
)
REQUIRED_QML_ROOTS = ("ui/pyside_app/qml", "ui/qml")
QT_PLATFORM_CANDIDATES = (
    "PySide6/Qt/plugins/platforms/qwindows.dll",
    "PySide6/Qt6/plugins/platforms/qwindows.dll",
    "platforms/qwindows.dll",
)


@dataclass(slots=True)
class ArtifactValidationResult:
    artifact_dir: str
    data_root: str
    ok: bool
    missing: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "artifact_dir": self.artifact_dir,
                "data_root": self.data_root,
                "ok": self.ok,
                "missing": self.missing,
                "forbidden": self.forbidden,
            },
            indent=2,
            sort_keys=True,
        )


def artifact_data_root(artifact_root: Path) -> Path:
    """Return the authoritative PyInstaller data root for an onedir artifact."""

    internal = artifact_root / "_internal"
    return internal if internal.is_dir() else artifact_root


def _as_posix_relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_forbidden(relative: str) -> bool:
    lowered = relative.lower()
    parts = [part.lower() for part in Path(relative).parts]
    if any(part in FORBIDDEN_PROJECT_DIRS for part in parts):
        return True
    if any(
        part == forbidden or forbidden in part for part in parts for forbidden in FORBIDDEN_PARTS
    ):
        return True
    return any(lowered.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES)


def validate_artifact(artifact_dir: str | Path) -> ArtifactValidationResult:
    root = Path(artifact_dir).resolve()
    data_root = artifact_data_root(root)
    missing: list[str] = []
    forbidden: list[str] = []
    if not root.exists() or not root.is_dir():
        return ArtifactValidationResult(
            root.as_posix(), data_root.as_posix(), False, missing=["artifact_dir"]
        )

    for path in sorted(root.rglob("*")):
        relative = _as_posix_relative(root, path)
        if _is_forbidden(relative):
            forbidden.append(relative)

    for relative in REQUIRED_ROOT_FILES:
        if not (root / relative).is_file():
            missing.append(relative)
    for relative in REQUIRED_DATA_FILES:
        if not (data_root / relative).is_file():
            missing.append(relative)
    for relative in REQUIRED_QML_ROOTS:
        if not (data_root / relative).is_dir():
            missing.append(relative)
    if not any((data_root / relative).is_file() for relative in QT_PLATFORM_CANDIDATES):
        missing.append("Qt qwindows.dll platform plugin")

    return ArtifactValidationResult(
        root.as_posix(), data_root.as_posix(), not missing and not forbidden, missing, forbidden
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate GymOS Windows onedir artifact contents.")
    parser.add_argument("artifact_dir", nargs="?", default="build/output/GymOS")
    args = parser.parse_args(argv)
    result = validate_artifact(args.artifact_dir)
    print(result.to_json())
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
