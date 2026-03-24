from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_installer_module(repo_root: Path, monkeypatch):
    deploy_module = types.ModuleType("deploy")
    packaging_module = types.ModuleType("deploy.packaging")
    packaging_module.build_pyinstaller_bundle = object()
    deploy_module.packaging = packaging_module

    scripts_module = types.ModuleType("scripts")
    scripts_module.oem_provision_license = object()

    monkeypatch.setitem(sys.modules, "deploy", deploy_module)
    monkeypatch.setitem(sys.modules, "deploy.packaging", packaging_module)
    monkeypatch.setitem(sys.modules, "scripts", scripts_module)

    module_path = repo_root / "scripts" / "build_installer_from_profile.py"
    module_name = "installer_profile_module_for_tests"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _normalized(path: Path) -> Path:
    return Path(str(path).replace("\\", "/")).resolve()


def test_read_profile_windows_paths_resolve_to_repo_root(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    installer = _load_installer_module(repo_root, monkeypatch)
    profile_path = repo_root / "deploy" / "packaging" / "profiles" / "windows.toml"

    profile = installer._read_profile(profile_path)

    assert profile.pyinstaller is not None
    assert (
        _normalized(profile.pyinstaller.entrypoint).as_posix().endswith("scripts/run_local_bot.py")
    )
    assert _normalized(profile.pyinstaller.entrypoint) == repo_root / "scripts" / "run_local_bot.py"
    assert _normalized(profile.pyinstaller.dist_dir) == (
        repo_root / "var" / "dist" / "pyinstaller" / "windows"
    )
    assert _normalized(profile.pyinstaller.work_dir) == (
        repo_root / "var" / "build" / "pyinstaller" / "windows"
    )

    assert profile.briefcase is not None
    assert _normalized(profile.briefcase.project_path) == repo_root / "ui" / "briefcase"
    assert _normalized(profile.briefcase.output_dir) == (
        repo_root / "var" / "dist" / "briefcase" / "windows"
    )

    assert (
        _normalized(profile.bundle.output_dir)
        == repo_root / "var" / "dist" / "installers" / "windows"
    )
    assert (
        _normalized(profile.bundle.work_dir)
        == repo_root / "var" / "build" / "installers" / "windows"
    )
    assert _normalized(profile.bundle.metadata_path) == (
        repo_root / "var" / "dist" / "installers" / "windows" / "installer_metadata.json"
    )


def test_build_pyinstaller_uses_runtime_name_for_expected_artifact(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    installer = _load_installer_module(repo_root, monkeypatch)

    dist_dir = tmp_path / "var" / "dist" / "pyinstaller" / "windows"
    expected = dist_dir / "bot_core_runtime" / "bot_core_runtime.exe"
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_text("exe", encoding="utf-8")

    subprocess_calls: list[list[str]] = []

    def _fake_run(args, check):
        subprocess_calls.append(args)
        assert check is True

    monkeypatch.setattr(installer.subprocess, "run", _fake_run)

    profile = installer.PyInstallerProfile(
        entrypoint=tmp_path / "scripts" / "run_local_bot.py",
        runtime_name="bot_core_runtime",
        hidden_imports=(),
        dist_dir=dist_dir,
        work_dir=None,
    )

    candidate = installer._build_pyinstaller(profile, "windows")

    assert subprocess_calls
    assert "--name" in subprocess_calls[0]
    assert (
        str(expected)
        .replace("\\", "/")
        .endswith("var/dist/pyinstaller/windows/bot_core_runtime/bot_core_runtime.exe")
    )
    assert (
        str(candidate)
        .replace("\\", "/")
        .endswith("var/dist/pyinstaller/windows/bot_core_runtime/bot_core_runtime.exe")
    )
    assert "run_local_bot/run_local_bot.exe" not in str(candidate).replace("\\", "/")


def test_main_blocks_windows_pyinstaller_on_non_windows_host(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    installer = _load_installer_module(repo_root, monkeypatch)

    profile_file = tmp_path / "windows.toml"
    profile_file.write_text("platform='windows'\n", encoding="utf-8")

    pyinstaller_profile = installer.PyInstallerProfile(
        entrypoint=tmp_path / "scripts" / "run_local_bot.py",
        runtime_name="bot_core_runtime",
        hidden_imports=(),
        dist_dir=tmp_path / "dist",
        work_dir=tmp_path / "work",
    )
    bundle_profile = installer.BundleProfile(
        output_dir=tmp_path / "out",
        work_dir=tmp_path / "work-installer",
        qt_dist=None,
        include=(),
        metadata_path=tmp_path / "metadata.json",
        signing_key=None,
        signing_key_id=None,
    )
    profile = installer.Profile(
        platform="windows",
        pyinstaller=pyinstaller_profile,
        briefcase=None,
        bundle=bundle_profile,
    )

    monkeypatch.setattr(installer, "_read_profile", lambda _: profile)
    monkeypatch.setattr(installer.os, "name", "posix", raising=False)

    build_called = False

    def _unexpected_build(*_args, **_kwargs):
        nonlocal build_called
        build_called = True
        raise AssertionError("_build_pyinstaller should not be called on non-Windows host")

    monkeypatch.setattr(installer, "_build_pyinstaller", _unexpected_build)

    with pytest.raises(SystemExit, match="wymaga uruchomienia PyInstaller na Windows"):
        installer.main(
            ["--profile", str(profile_file), "--version", "1.2.3", "--platform", "windows"]
        )

    assert build_called is False
