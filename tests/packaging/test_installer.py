from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from deploy.packaging.desktop_installer import DesktopInstallerBuilder


def _write_linux_profile(tmp_path: Path, repo_root: Path) -> Path:
    profiles_dir = tmp_path / "profiles" / "demo"
    profiles_dir.mkdir(parents=True)

    samples = repo_root / "deploy" / "packaging" / "assets" / "demo"
    output_dir = tmp_path / "dist"
    work_dir = tmp_path / "build"
    metadata_path = tmp_path / "metadata" / "linux.json"

    profile = f"""
platform = "linux"

[bundle]
output_dir = "{output_dir.as_posix()}"
work_dir = "{work_dir.as_posix()}"
qt_dist = "{(samples / "ui").as_posix()}"
include = [
  "config={(samples / "config").as_posix()}",
  "daemon={(samples / "daemon").as_posix()}",
  "resources={(samples / "resources").as_posix()}",
  "reports={(samples / "reports").as_posix()}",
  "var={(samples / "var").as_posix()}",
]
wheels_extra = [
  "{(samples / "wheels" / "ccxt-4.0.0-py3-none-any.whl").as_posix()}",
  "{(samples / "wheels" / "aiohttp-3.9.5-cp311-cp311-manylinux.whl").as_posix()}",
  "{(samples / "wheels" / "lightgbm-4.6.0-cp311-cp311-manylinux.whl").as_posix()}",
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
    assert "wheels/ccxt-4.0.0-py3-none-any.whl" in members
    assert "reports/champion_overview.json" in members
    assert "var/models/quality/sample_model/champion.json" in members


def test_validate_hook_executes_hwid_check(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    profiles_dir = _write_linux_profile(tmp_path, repo_root)

    builder = DesktopInstallerBuilder(
        version="2.0.0",
        profiles_dir=profiles_dir,
        hwid_hook_source=repo_root / "probe_keyring.py",
    )

    archive_path = builder.build("linux")

    extract_dir = tmp_path / "extract"
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extract_dir)

    hook_path = extract_dir / "hooks" / "validate_hwid.py"
    assert hook_path.exists()

    # Środowisko testowe nie posiada zainstalowanej biblioteki ``keyring``.
    # Dołączamy lekki stub w katalogu hooków, aby symulować obecność modułu
    # podczas uruchamiania walidatora HWID.
    (extract_dir / "hooks" / "keyring.py").write_text(
        "_STORE = {}\n\n"
        "class errors:\n"
        "    class KeyringError(Exception):\n"
        "        ...\n\n"
        "def get_password(service, username):\n"
        "    return _STORE.get((service, username))\n\n"
        "def set_password(service, username, secret):\n"
        "    _STORE[(service, username)] = secret\n\n"
        "def delete_password(service, username):\n"
        "    _STORE.pop((service, username), None)\n\n"
        "backend = None\n",
        encoding="utf-8",
    )

    expected_path = tmp_path / "expected.json"
    fingerprint = "fake-hwid-200"
    expected_path.write_text(json.dumps({"fingerprint": fingerprint}), encoding="utf-8")

    log_path = tmp_path / "hook.log"

    env = os.environ.copy()
    env.update(
        {
            "KBOT_EXPECTED_HWID_FILE": str(expected_path),
            "KBOT_INSTALL_LOG": str(log_path),
            "KBOT_FAKE_FINGERPRINT": fingerprint,
        }
    )

    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        os.pathsep.join([str(repo_root), python_path]) if python_path else str(repo_root)
    )

    result = subprocess.run(
        [sys.executable, str(hook_path)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise AssertionError(
            "Hook walidacji HWID zakończył się błędem", result.stderr or result.stdout
        )

    assert log_path.exists(), "Hook powinien wygenerować log walidacji"
    log_text = log_path.read_text(encoding="utf-8")
    assert fingerprint in log_text


def test_prod_profile_fails_fast_when_assets_are_placeholders(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    builder = DesktopInstallerBuilder(
        version="9.9.9",
        profiles_dir=repo_root / "deploy" / "packaging" / "profiles",
        hwid_hook_source=repo_root / "probe_keyring.py",
    )

    with pytest.raises(SystemExit, match="nieprovisionowany katalog"):
        builder.build("linux")


def test_prod_profile_rejects_demo_assets_path(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True)
    demo_assets = repo_root / "deploy" / "packaging" / "assets" / "demo"

    profile = f"""
platform = "linux"

[bundle]
output_dir = "{(tmp_path / "out").as_posix()}"
work_dir = "{(tmp_path / "work").as_posix()}"
qt_dist = "{(demo_assets / "ui").as_posix()}"
include = [
  "config={(demo_assets / "config").as_posix()}",
]
wheels_extra = []
metadata_path = "{(tmp_path / "metadata.json").as_posix()}"
"""
    (profiles_dir / "linux.toml").write_text(profile.strip() + "\n", encoding="utf-8")

    builder = DesktopInstallerBuilder(
        version="1.0.0",
        profiles_dir=profiles_dir,
        hwid_hook_source=repo_root / "probe_keyring.py",
    )

    with pytest.raises(SystemExit, match="demo"):
        builder._load_profile("linux")


def test_prod_profile_rejects_demo_wheels_extra(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True)
    demo_assets = repo_root / "deploy" / "packaging" / "assets" / "demo"
    external_assets = tmp_path / "external"
    external_assets.mkdir(parents=True)

    profile = f"""
platform = "linux"

[bundle]
output_dir = "{(tmp_path / "out").as_posix()}"
work_dir = "{(tmp_path / "work").as_posix()}"
qt_dist = "{external_assets.as_posix()}"
include = [
  "config={external_assets.as_posix()}",
]
wheels_extra = [
  "{(demo_assets / "wheels" / "ccxt-4.0.0-py3-none-any.whl").as_posix()}",
]
metadata_path = "{(tmp_path / "metadata.json").as_posix()}"
"""
    (profiles_dir / "linux.toml").write_text(profile.strip() + "\n", encoding="utf-8")

    builder = DesktopInstallerBuilder(
        version="1.0.0",
        profiles_dir=profiles_dir,
        hwid_hook_source=repo_root / "probe_keyring.py",
    )

    with pytest.raises(SystemExit, match="demo"):
        builder._load_profile("linux")


def test_non_demo_profile_allows_assets_outside_demo_and_prod(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True)

    external_root = tmp_path / "external_assets"
    qt_dir = external_root / "ui"
    config_dir = external_root / "config"
    wheel_path = external_root / "wheels" / "external-1.0.0-py3-none-any.whl"
    qt_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    wheel_path.parent.mkdir(parents=True, exist_ok=True)
    wheel_path.write_text("wheel", encoding="utf-8")

    profile = f"""
platform = "linux"

[bundle]
output_dir = "{(tmp_path / "out").as_posix()}"
work_dir = "{(tmp_path / "work").as_posix()}"
qt_dist = "{qt_dir.as_posix()}"
include = [
  "config={config_dir.as_posix()}",
]
wheels_extra = [
  "{wheel_path.as_posix()}",
]
metadata_path = "{(tmp_path / "metadata.json").as_posix()}"
"""
    (profiles_dir / "linux.toml").write_text(profile.strip() + "\n", encoding="utf-8")

    builder = DesktopInstallerBuilder(
        version="1.0.0",
        profiles_dir=profiles_dir,
        hwid_hook_source=repo_root / "probe_keyring.py",
    )

    config = builder._load_profile("linux")

    assert config.qt_dist == qt_dir.resolve()
    assert config.includes[0].source == config_dir.resolve()
    assert config.wheels == (wheel_path.resolve(),)


def test_demo_profile_keeps_demo_assets_explicit(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    builder = DesktopInstallerBuilder(
        version="1.0.0",
        profiles_dir=repo_root / "deploy" / "packaging" / "profiles" / "demo",
        hwid_hook_source=repo_root / "probe_keyring.py",
    )

    config = builder._load_profile("linux")

    assert config.qt_dist is not None
    assert "/assets/demo/" in config.qt_dist.as_posix().replace("\\", "/")
    assert all(
        "/assets/demo/" in spec.source.as_posix().replace("\\", "/") for spec in config.includes
    )


def test_invalid_profile_does_not_fallback_to_demo_defaults(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "linux.toml").write_text(
        """
platform = "linux"

[bundle]
output_dir = "var/dist/installers"
work_dir = "var/build/installers"
qt_dist = "../assets/demo/ui"
include = [
  "broken-entry-without-separator"
]
metadata_path = "var/dist/installers/installer_metadata.json"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    builder = DesktopInstallerBuilder(
        version="1.0.0",
        profiles_dir=profiles_dir,
        hwid_hook_source=repo_root / "probe_keyring.py",
    )

    with pytest.raises(SystemExit, match="include"):
        builder._load_profile("linux")
