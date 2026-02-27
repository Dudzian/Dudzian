from __future__ import annotations

import re
from pathlib import Path

QML_ROOT = Path("ui/pyside_app/qml")
LOWERCASE_STYLES_DIR = QML_ROOT / "styles"

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
    if not LOWERCASE_STYLES_DIR.exists():
        return

    qml_files = sorted(p.relative_to(QML_ROOT).as_posix() for p in LOWERCASE_STYLES_DIR.rglob("*.qml"))
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
