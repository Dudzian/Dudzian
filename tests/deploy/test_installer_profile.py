from __future__ import annotations

from pathlib import Path

import pytest

from deploy.packaging.installer_profile import normalize_profile_path, read_profile


def test_normalize_profile_path_accepts_windows_separators(tmp_path: Path) -> None:
    base = tmp_path / "profiles"
    base.mkdir(parents=True)

    resolved = normalize_profile_path(r"..\scripts\run_local_bot.py", base_dir=base)

    assert resolved == (tmp_path / "scripts" / "run_local_bot.py").resolve()


def test_read_profile_parses_complete_profile(tmp_path: Path) -> None:
    profile_path = tmp_path / "installer.toml"
    profile_path.write_text(
        """
platform = "linux"

[pyinstaller]
entrypoint = "scripts/run_local_bot.py"
runtime_name = "bot_core_runtime"
hidden_imports = ["a", "b"]
dist_dir = "dist/pyinstaller"
work_dir = "build/pyinstaller"

[briefcase]
project = "ui/briefcase"
app = "Dudzian"
output_dir = "dist/briefcase"

[bundle]
output_dir = "dist/installers"
work_dir = "build/installers"
qt_dist = "qt"
include = ["config=assets/prod/config"]
metadata_path = "dist/installers/meta.json"
signing_key = "secret"
signing_key_id = "key-1"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    profile = read_profile(profile_path)

    assert profile.platform == "linux"
    assert profile.pyinstaller is not None
    assert profile.pyinstaller.runtime_name == "bot_core_runtime"
    assert profile.pyinstaller.hidden_imports == ("a", "b")
    assert profile.pyinstaller.entrypoint == (tmp_path / "scripts" / "run_local_bot.py").resolve()
    assert profile.briefcase is not None
    assert profile.briefcase.project_path == (tmp_path / "ui" / "briefcase").resolve()
    assert profile.bundle.output_dir == (tmp_path / "dist" / "installers").resolve()
    assert profile.bundle.metadata_path == (tmp_path / "dist" / "installers" / "meta.json").resolve()


def test_read_profile_returns_none_for_optional_sections(tmp_path: Path) -> None:
    profile_path = tmp_path / "minimal.toml"
    profile_path.write_text(
        """
platform = "linux"

[bundle]
output_dir = "dist/installers"
work_dir = "build/installers"
metadata_path = "dist/installers/meta.json"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    profile = read_profile(profile_path)

    assert profile.pyinstaller is None
    assert profile.briefcase is None


def test_read_profile_requires_platform(tmp_path: Path) -> None:
    profile_path = tmp_path / "invalid.toml"
    profile_path.write_text(
        """
[bundle]
output_dir = "dist/installers"
work_dir = "build/installers"
metadata_path = "dist/installers/meta.json"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="platform"):
        read_profile(profile_path)



def test_read_profile_requires_bundle_metadata_path(tmp_path: Path) -> None:
    profile_path = tmp_path / "missing_metadata.toml"
    profile_path.write_text(
        """
platform = "linux"

[bundle]
output_dir = "dist/installers"
work_dir = "build/installers"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="metadata_path"):
        read_profile(profile_path)


def test_read_profile_requires_bundle_output_and_work_dir(tmp_path: Path) -> None:
    profile_path = tmp_path / "missing_bundle_paths.toml"
    profile_path.write_text(
        """
platform = "linux"

[bundle]
metadata_path = "dist/installers/meta.json"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="output_dir"):
        read_profile(profile_path)



def test_read_profile_requires_bundle_work_dir(tmp_path: Path) -> None:
    profile_path = tmp_path / "missing_work_dir.toml"
    profile_path.write_text(
        """
platform = "linux"

[bundle]
output_dir = "dist/installers"
metadata_path = "dist/installers/meta.json"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="work_dir"):
        read_profile(profile_path)
