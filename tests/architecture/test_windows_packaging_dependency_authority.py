"""Architecture contract for exact native Windows packaging dependencies."""

from pathlib import Path

from scripts.ci.bootstrap_dependency import locked_versions

ROOT = Path(__file__).parents[2]
WINDOWS_RUNTIME_INSTALL = (
    "python -m pip install --no-deps -r deploy/packaging/requirements-windows-runtime.lock"
)


def _job(path: str, start: str, end: str | None = None) -> str:
    text = (ROOT / path).read_text(encoding="utf-8")
    body = text.split(start, 1)[1]
    return body.split(end, 1)[0] if end else body


def test_all_canonical_windows_packaging_authorities_install_runtime_lock() -> None:
    authorities = {
        ".github/workflows/windows-build.yml": ("  build-windows:", None),
        ".github/workflows/main.yml": (
            "  ui-packaging-windows:",
            "\n  ui-packaging-macos:",
        ),
        "deploy/ci/github_actions_cross_installer.yml": (
            "      - name: Install packaging requirements (Windows)",
            "\n      - name: Validate marketplace presets (Windows)",
        ),
        ".github/workflows/platform-deployment.yml": (
            "  windows-clean-install-integration:",
            "\n  windows-stage10-prepare-and-schedule-reboot:",
        ),
    }
    for path, boundaries in authorities.items():
        section = _job(path, *boundaries)
        assert "requirements-desktop.lock" in section, path
        assert section.count(WINDOWS_RUNTIME_INSTALL) == 1, path
        assert "pip install pywin32" not in section, path


def test_windows_runtime_lock_is_an_exact_closed_contract() -> None:
    lock = ROOT / "deploy/packaging/requirements-windows-runtime.lock"
    assert locked_versions(lock) == {"pywin32": "312"}
