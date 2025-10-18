"""Builds an offline installer bundle for the Qt desktop shell."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from bot_core.security.signing import build_hmac_signature, canonical_json_bytes


DEFAULT_OUTPUT = Path("var/dist/desktop")
DEFAULT_REPORTS_DIR = Path("var/reports")


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _run_pyinstaller(entrypoint: Path, workdir: Path, platform: str) -> Path:
    output_dir = workdir / "pyinstaller"
    build_dir = output_dir / "build"
    dist_dir = output_dir / "dist"
    output_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "pyinstaller",
        "--clean",
        "--noconfirm",
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(build_dir),
        str(entrypoint),
    ]
    subprocess.run(args, check=True)
    extension = ".exe" if platform == "windows" else ""
    executable = dist_dir / entrypoint.stem / f"{entrypoint.stem}{extension}"
    if not executable.exists():
        raise RuntimeError(f"PyInstaller nie zbudował updatera w {executable}")
    return executable


def _sign_updater(updater: Path, key: str, destination: Path) -> Path:
    payload = {
        "path": str(updater.resolve()),
        "size": updater.stat().st_size,
    }
    signature = build_hmac_signature(key.encode("utf-8"), canonical_json_bytes(payload))
    destination.write_text(signature["value"], encoding="utf-8")
    return destination


def build_bundle(args: argparse.Namespace) -> Path:
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    bundle_dir = output_root / "bot_trading_shell"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    build_dir = Path(args.build_dir).expanduser().resolve()
    binary_name = "bot_trading_shell.exe" if args.platform == "windows" else "bot_trading_shell"
    binary_path = build_dir / binary_name
    if not binary_path.exists():
        raise SystemExit(f"Nie znaleziono skompilowanej aplikacji pod {binary_path}")

    shutil.copy2(binary_path, bundle_dir / binary_name)
    _copy_tree(Path("ui/qml"), bundle_dir / "qml")
    _copy_tree(Path("config"), bundle_dir / "config")
    _copy_tree(Path("bot_core"), bundle_dir / "bot_core")

    reports_dir = Path(args.reports).expanduser()
    if reports_dir.exists():
        _copy_tree(reports_dir, bundle_dir / "reports")

    updater_entry = Path(args.updater_script).expanduser().resolve()
    updater_binary = _run_pyinstaller(updater_entry, output_root, args.platform)
    updater_dest = bundle_dir / updater_binary.name
    shutil.copy2(updater_binary, updater_dest)

    if args.signing_key:
        signature_path = bundle_dir / f"{updater_binary.stem}.sig"
        _sign_updater(updater_dest, args.signing_key, signature_path)

    manifest = {
        "bundle": binary_name,
        "updater": updater_dest.name,
        "has_signature": bool(args.signing_key),
    }
    (bundle_dir / "INSTALL_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    archive_path = output_root / "bot_trading_shell_bundle.zip"
    if archive_path.exists():
        archive_path.unlink()
    shutil.make_archive(str(archive_path.with_suffix("")), "zip", bundle_dir)
    return archive_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build OEM desktop installer bundle")
    parser.add_argument("--build-dir", required=True, help="Path to compiled Qt application (Release directory)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output directory for the installer bundle")
    parser.add_argument("--reports", default=str(DEFAULT_REPORTS_DIR), help="Optional reports directory to embed")
    parser.add_argument("--updater-script", default="scripts/desktop_updater.py", help="Updater entrypoint for PyInstaller")
    parser.add_argument("--signing-key", help="Secret used to sign the updater binary")
    parser.add_argument("--platform", choices=["linux", "windows", "mac"], default="linux")
    args = parser.parse_args(argv)

    try:
        archive = build_bundle(args)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"PyInstaller zakończył się błędem: {exc}") from exc
    except Exception as exc:  # pragma: no cover - CLI diagnostics
        raise SystemExit(str(exc)) from exc

    print(f"Zbudowano pakiet instalacyjny: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
