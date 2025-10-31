from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from deploy.packaging.desktop_installer import DesktopInstallerBuilder


def _write_linux_profile(tmp_path: Path, repo_root: Path) -> Path:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True)

    samples = repo_root / "deploy" / "packaging" / "samples"
    output_dir = tmp_path / "dist"
    work_dir = tmp_path / "build"
    metadata_path = tmp_path / "metadata" / "linux.json"

    profile = f"""
platform = "linux"

[bundle]
output_dir = "{output_dir.as_posix()}"
work_dir = "{work_dir.as_posix()}"
qt_dist = "{(samples / 'ui').as_posix()}"
include = [
  "config={(samples / 'config').as_posix()}",
  "daemon={(samples / 'daemon').as_posix()}",
  "resources={(samples / 'resources').as_posix()}",
]
metadata_path = "{metadata_path.as_posix()}"
"""
    (profiles_dir / "linux.toml").write_text(profile.strip() + "\n", encoding="utf-8")
    return profiles_dir


@pytest.mark.parametrize("platform", ["linux"])
def test_builder_creates_archive_with_hwid_hook(tmp_path: Path, platform: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    profiles_dir = _write_linux_profile(tmp_path, repo_root)

    builder = DesktopInstallerBuilder(
        version="1.2.3",
        profiles_dir=profiles_dir,
        hwid_hook_source=repo_root / "probe_keyring.py",
    )

    archive_path = builder.build(platform)

    assert archive_path.exists()

    metadata_path = tmp_path / "metadata" / f"{platform}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["platform"] == platform
    assert metadata["archive"] == archive_path.name
    assert metadata["hwid_validation"]["probe"] == "hooks/probe_keyring.py"

    with zipfile.ZipFile(archive_path) as zf:
        members = set(zf.namelist())

    assert "hooks/validate_hwid.py" in members
    assert "hooks/probe_keyring.py" in members
    assert "manifest.json" in members
    assert any(name.startswith("config/") for name in members)
