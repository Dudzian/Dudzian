import argparse
import json
import zipfile
from pathlib import Path

import pytest

from scripts import build_desktop_installer
from bot_core.security.signing import build_hmac_signature


def test_build_bundle_creates_signed_archive(monkeypatch, tmp_path):
    build_dir = tmp_path / "qt_build"
    build_dir.mkdir()
    binary = build_dir / "bot_trading_shell"
    binary.write_bytes(b"binary")

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "daily.json").write_text("{}", encoding="utf-8")

    updater_entry = tmp_path / "desktop_updater.py"
    updater_entry.write_text("print('hi')\n", encoding="utf-8")

    output_root = tmp_path / "dist"
    args = argparse.Namespace(
        build_dir=str(build_dir),
        output=str(output_root),
        reports=str(reports_dir),
        updater_script=str(updater_entry),
        signing_key="super-secret",
        platform="linux",
    )

    expected_output_root = Path(args.output).expanduser().resolve()
    fake_updater_binary = tmp_path / "desktop_updater"
    fake_updater_binary.write_bytes(b"updater")

    copied_directories = []

    def fake_copy_tree(source: Path, destination: Path) -> None:
        copied_directories.append((source, destination))
        destination.mkdir(parents=True, exist_ok=True)
        placeholder = destination / f"{source.name}.placeholder"
        placeholder.write_text(source.name, encoding="utf-8")

    def fake_run_pyinstaller(entrypoint: Path, workdir: Path, platform: str) -> Path:
        assert entrypoint == updater_entry.resolve()
        assert workdir == expected_output_root
        assert platform == args.platform
        return fake_updater_binary

    monkeypatch.setattr(build_desktop_installer, "_copy_tree", fake_copy_tree)
    monkeypatch.setattr(build_desktop_installer, "_run_pyinstaller", fake_run_pyinstaller)

    archive_path = build_desktop_installer.build_bundle(args)

    assert archive_path == expected_output_root / "bot_trading_shell_bundle.zip"
    assert archive_path.exists()

    bundle_dir = expected_output_root / "bot_trading_shell"
    updater_dest = bundle_dir / "desktop_updater"
    signature_value = (bundle_dir / "desktop_updater.sig").read_text(encoding="utf-8").strip()
    expected_signature = build_hmac_signature(
        {
            "path": str(updater_dest.resolve()),
            "size": updater_dest.stat().st_size,
        },
        key=args.signing_key.encode("utf-8"),
    )
    assert signature_value == expected_signature["value"]

    # ensure kopiowane katalogi zostały odwzorowane w strukturze bundla
    copied_sources = {str(source) for source, _ in copied_directories}
    assert {"ui/qml", "config", "bot_core", str(reports_dir)}.issubset(copied_sources)

    with zipfile.ZipFile(archive_path) as bundle:
        members = set(bundle.namelist())
        assert "bot_trading_shell" in members
        assert "desktop_updater" in members
        assert "desktop_updater.sig" in members
        assert "INSTALL_MANIFEST.json" in members

        manifest = json.loads(bundle.read("INSTALL_MANIFEST.json"))
        assert manifest["bundle"] == "bot_trading_shell"
        assert manifest["updater"] == "desktop_updater"
        assert manifest["has_signature"] is True


def test_build_bundle_requires_compiled_binary(tmp_path):
    build_dir = tmp_path / "missing_build"
    build_dir.mkdir()

    args = argparse.Namespace(
        build_dir=str(build_dir),
        output=str(tmp_path / "out"),
        reports=str(tmp_path / "reports"),
        updater_script=str(tmp_path / "desktop_updater.py"),
        signing_key=None,
        platform="linux",
    )

    with pytest.raises(SystemExit) as excinfo:
        build_desktop_installer.build_bundle(args)

    assert "Nie znaleziono skompilowanej aplikacji" in str(excinfo.value)
