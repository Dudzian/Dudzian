#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
PACKAGING_DIR = ROOT / "deploy" / "packaging" / "profiles"

DEFAULT_PROFILES = {
    "linux": PACKAGING_DIR / "linux.toml",
    "macos": PACKAGING_DIR / "macos.toml",
    "windows": PACKAGING_DIR / "windows.toml",
}


def detect_powershell() -> str | None:
    for candidate in ("pwsh", "powershell", "powershell.exe"):
        if shutil.which(candidate):
            return candidate
    return None


def run_command(command: List[str], *, env: Dict[str, str]) -> None:
    print(f"[build] running: {' '.join(command)}")
    subprocess.run(command, check=True, env=env)


def build_linux(profile: Path, version: str, output_dir: Path, extra_args: List[str]) -> None:
    script = SCRIPTS_DIR / "build_installer_linux.sh"
    env = os.environ.copy()
    env["PROFILE"] = str(profile)
    env["VERSION"] = version
    env["OUTPUT_DIR"] = str(output_dir)
    run_command(["bash", str(script), *extra_args], env=env)


def build_macos(profile: Path, version: str, output_dir: Path, extra_args: List[str]) -> None:
    script = SCRIPTS_DIR / "build_installer_macos.sh"
    env = os.environ.copy()
    env["PROFILE"] = str(profile)
    env["VERSION"] = version
    env["OUTPUT_DIR"] = str(output_dir)
    run_command(["bash", str(script), *extra_args], env=env)


def build_windows(profile: Path, version: str, output_dir: Path, extra_args: List[str]) -> None:
    powershell = detect_powershell()
    env = os.environ.copy()
    env["PROFILE"] = str(profile)
    env["VERSION"] = version
    env["OUTPUT_DIR"] = str(output_dir)

    ps_script = SCRIPTS_DIR / "build_installer_windows.ps1"
    sh_script = SCRIPTS_DIR / "build_installer_windows.sh"

    if platform.system().lower().startswith("win") and powershell:
        run_command([powershell, "-File", str(ps_script), *extra_args], env=env)
    else:
        run_command(["bash", str(sh_script), *extra_args], env=env)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build installers for all supported platforms")
    parser.add_argument("--version", default="0.0.0-dev", help="Override installer version tag.")
    parser.add_argument("--output-dir", default=str(ROOT / "dist"), help="Destination directory for artifacts.")
    parser.add_argument("--profile-linux", default=str(DEFAULT_PROFILES["linux"]), help="Custom Linux profile path.")
    parser.add_argument("--profile-macos", default=str(DEFAULT_PROFILES["macos"]), help="Custom macOS profile path.")
    parser.add_argument("--profile-windows", default=str(DEFAULT_PROFILES["windows"]), help="Custom Windows profile path.")
    parser.add_argument("--skip-linux", action="store_true", help="Skip building the Linux installer.")
    parser.add_argument("--skip-macos", action="store_true", help="Skip building the macOS installer.")
    parser.add_argument("--skip-windows", action="store_true", help="Skip building the Windows installer.")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Additional arguments passed to underlying scripts.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    version = args.version
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    extra_args = [arg for arg in args.extra if arg]

    if not args.skip_linux:
        build_linux(Path(args.profile_linux).expanduser(), version, output_dir, extra_args)

    if not args.skip_macos:
        build_macos(Path(args.profile_macos).expanduser(), version, output_dir, extra_args)

    if not args.skip_windows:
        build_windows(Path(args.profile_windows).expanduser(), version, output_dir, extra_args)

    print(f"[build] installers stored in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
