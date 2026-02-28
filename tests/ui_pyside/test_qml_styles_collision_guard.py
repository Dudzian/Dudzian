from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

QML_ROOT = Path("ui/pyside_app/qml")

# Match only path-based imports like:
# - import "styles" as Styles
# - import "../styles" as Styles
# - import "styles\\DesignSystem.qml" as Styles
# - matches both ../styles and ../Styles (IGNORECASE)
# - does not match module imports like: import Styles 1.0 as StylesModule
# Path-based imports are the risky part in QML because they can bypass module resolution
# and trigger case-insensitive shadowing/collisions on Windows filesystems.
PATH_STYLES_IMPORT_RE = re.compile(
    r'^\s*import\s+"(?P<path>(?:styles|(?:[^"]*[\\/])styles)(?:[\\/][^"]*)?)"\s+as\s+\w+',
    flags=re.IGNORECASE,
)


def test_no_lowercase_styles_qml_wrapper_files() -> None:
    """Lowercase styles QML wrappers can collide with Styles/ on case-insensitive FS."""
    # Prefer git index view (case-accurate) over filesystem (can be case-weird on
    # Windows checkouts after a rename between styles <-> Styles).
    if shutil.which("git"):
        try:
            out = subprocess.check_output(
                ["git", "ls-tree", "-r", "--name-only", "HEAD", "--", QML_ROOT.as_posix()],
                text=True,
                stderr=subprocess.STDOUT,
            )
            offenders = sorted(
                path
                for path in out.splitlines()
                if path.startswith(f"{QML_ROOT.as_posix()}/styles/") and path.endswith(".qml")
            )
            assert offenders == [], (
                "Case-insensitive collision risk: git tree contains lowercase styles QML files:\n"
                + "\n".join(offenders)
            )
            return
        except Exception:
            # Fall through to filesystem checks below if git metadata is unavailable.
            pass

    # Fallback: on Windows, filesystem casing can be unstable after renames and may
    # misreport a canonical Styles/ checkout as styles/. Skip in that scenario.
    if sys.platform == "win32":
        pytest.skip(
            "Filesystem casing on Windows can be unstable after renames; guarded via git-tree when available."
        )

    lowercase_styles_dir = QML_ROOT / "styles"
    if not lowercase_styles_dir.exists():
        return

    qml_files = sorted(path.relative_to(QML_ROOT).as_posix() for path in lowercase_styles_dir.rglob("*.qml"))
    assert qml_files == [], (
        "Case-insensitive collision risk: lowercase 'ui/pyside_app/qml/styles' contains QML files: "
        + ", ".join(qml_files)
    )


def test_no_path_based_styles_imports_in_pyside_qml() -> None:
    offenders: list[str] = []

    for qml_file in sorted(QML_ROOT.rglob("*.qml")):
        for line_number, line in enumerate(qml_file.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if PATH_STYLES_IMPORT_RE.match(line):
                offenders.append(f"{qml_file}:{line_number}: {line.rstrip()}")

    assert offenders == [], (
        "Found case-insensitive collision risk in path-based styles imports (file:line:import):\n"
        + "\n".join(offenders)
    )
