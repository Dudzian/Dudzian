#!/usr/bin/env python
"""Build a wheelhouse for offline installs."""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(wheelhouse: Path) -> None:
    manifest = wheelhouse / "manifest.txt"
    lines: list[str] = []
    for file in sorted(wheelhouse.glob("*.whl")):
        lines.append(f"{file.name}\tsha256={sha256sum(file)}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[wheelhouse] wrote manifest with {len(lines)} entries -> {manifest}")


def build_download_cmd(
    wheelhouse: Path,
    args: argparse.Namespace,
    packages: Iterable[str],
    python_executable: str,
) -> list[str]:
    cmd = [python_executable, "-m", "pip", "download", "--dest", str(wheelhouse)]
    if args.no_binary:
        cmd.extend(["--no-binary", args.no_binary])
    if args.index_url:
        cmd.extend(["--index-url", args.index_url])
    if args.extra_index_url:
        cmd.extend(["--extra-index-url", args.extra_index_url])
    if args.find_links:
        cmd.extend(["--find-links", args.find_links])
    if args.only_binary:
        cmd.extend(["--only-binary", args.only_binary])
    cmd.extend(packages)
    return cmd


def ensure_wheelhouse(wheelhouse: Path) -> None:
    wheelhouse.mkdir(parents=True, exist_ok=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build wheelhouse for offline installs")
    parser.add_argument("--wheelhouse", default="wheelhouse", help="Target wheelhouse directory")
    parser.add_argument("--pyside6-version", default=os.environ.get("PYSIDE6_VERSION", "6.7.0"))
    parser.add_argument("--requirements", help="Optional requirements file to download")
    parser.add_argument("--index-url", help="Primary index URL")
    parser.add_argument("--extra-index-url", help="Extra index URL")
    parser.add_argument("--find-links", help="Additional find-links for preloaded wheels")
    parser.add_argument("--only-binary", default=":all:", help="pip --only-binary value")
    parser.add_argument("--no-binary", help="pip --no-binary value")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    parser.add_argument("--skip-dev", action="store_true", help="Skip dev extras when downloading project deps")
    return parser.parse_args(argv)


def download(wheelhouse: Path, cmd: list[str]) -> None:
    print("[wheelhouse]", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Download failed with exit code {result.returncode}")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    wheelhouse = Path(args.wheelhouse).expanduser().resolve()
    ensure_wheelhouse(wheelhouse)

    # PySide6 stack
    pyside_packages = [
        f"PySide6=={args.pyside6_version}",
        f"PySide6_Addons=={args.pyside6_version}",
        f"PySide6_Essentials=={args.pyside6_version}",
        f"shiboken6=={args.pyside6_version}",
    ]
    download(wheelhouse, build_download_cmd(wheelhouse, args, pyside_packages, args.python))

    # PEP 517 build backend tooling required by pyproject.toml
    bootstrap_packages = ["wheel", "setuptools>=68"]
    download(wheelhouse, build_download_cmd(wheelhouse, args, bootstrap_packages, args.python))

    # Project dependencies
    project_target = ".[test]" if not args.skip_dev else "."
    download(wheelhouse, build_download_cmd(wheelhouse, args, [project_target], args.python))

    if args.requirements:
        download(wheelhouse, build_download_cmd(wheelhouse, args, [f"-r{args.requirements}"], args.python))

    write_manifest(wheelhouse)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
