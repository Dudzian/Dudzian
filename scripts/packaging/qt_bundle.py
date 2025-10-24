"""Helpery budujące i pakujące powłokę Qt na główne platformy."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)


def _run(command: Sequence[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    display = " ".join(command)
    logger.info("Uruchamiam: %s", display)
    final_env = os.environ.copy()
    if env:
        final_env.update(env)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=final_env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = (result.stdout or "").strip()
    if output:
        logger.debug("%s -- wyjście polecenia:%s%s", display, os.linesep, output)
    if result.returncode != 0:
        if output:
            logger.error("%s -- błąd podczas wykonywania:%s%s", display, os.linesep, output)
        raise RuntimeError(f"Polecenie zakończyło się kodem {result.returncode}: {display}")


def _detect_qt_prefix(candidate: str | None) -> str | None:
    if candidate:
        return candidate
    for key in ("QT_ROOT_DIR", "Qt6_DIR", "Qt_DIR", "QTDIR"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def configure_and_build(
    source_dir: Path,
    build_dir: Path,
    *,
    build_type: str = "Release",
    qt_prefix: str | None = None,
    install_prefix: Path | None = None,
    extra_cmake_args: Iterable[str] = (),
) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    cmake_args = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        f"-DCMAKE_BUILD_TYPE={build_type}",
    ]

    qt_candidate = _detect_qt_prefix(qt_prefix)
    if qt_candidate:
        cmake_args.append(f"-DCMAKE_PREFIX_PATH={qt_candidate}")
        qt6_dir = Path(qt_candidate) / "lib" / "cmake" / "Qt6"
        if qt6_dir.exists():
            cmake_args.append(f"-DQt6_DIR={qt6_dir}")

    cmake_args.extend(extra_cmake_args)
    _run(cmake_args)

    build_command = ["cmake", "--build", str(build_dir)]
    if build_type:
        build_command.extend(["--config", build_type])
    _run(build_command)

    if install_prefix is not None:
        install_prefix.mkdir(parents=True, exist_ok=True)
        install_command = ["cmake", "--install", str(build_dir), "--prefix", str(install_prefix)]
        if build_type:
            install_command.extend(["--config", build_type])
        _run(install_command)


def archive_bundle(install_dir: Path, output_dir: Path, *, bundle_name: str, platform_id: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_base = output_dir / f"{bundle_name}-{platform_id}"
    if platform_id == "windows":
        archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=install_dir))
    else:
        archive_path = Path(shutil.make_archive(str(archive_base), "gztar", root_dir=install_dir))
    logger.info("Utworzono archiwum: %s", archive_path)
    return archive_path


def _detect_platform(explicit: str) -> str:
    if explicit != "auto":
        return explicit
    current = sys.platform
    if current.startswith("win"):
        return "windows"
    if current == "darwin":
        return "macos"
    return "linux"


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("ui"), help="Katalog źródłowy projektu Qt")
    parser.add_argument("--build-dir", type=Path, default=Path("ui/build"), help="Katalog build")
    parser.add_argument("--install-dir", type=Path, default=None, help="Prefiks instalacji (opcjonalny)")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"), help="Katalog wynikowych archiwów")
    parser.add_argument("--bundle-name", default="bot_trading_shell", help="Bazowa nazwa bundla")
    parser.add_argument("--platform", choices=["auto", "linux", "windows", "macos"], default="auto")
    parser.add_argument("--build-type", default="Release", help="Typ builda CMake")
    parser.add_argument("--qt-prefix", default=None, help="Ścieżka do instalacji Qt (opcjonalna)")
    parser.add_argument("--extra-cmake", action="append", default=[], help="Dodatkowe argumenty cmake")
    parser.add_argument("--skip-archive", action="store_true", help="Nie tworzy archiwum")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = _parse_args(argv or sys.argv[1:])

    source_dir = args.source_dir.resolve()
    build_dir = args.build_dir.resolve()
    install_dir = args.install_dir.resolve() if args.install_dir else build_dir / "install"
    platform_id = _detect_platform(args.platform)

    configure_and_build(
        source_dir,
        build_dir,
        build_type=args.build_type,
        qt_prefix=args.qt_prefix,
        install_prefix=install_dir,
        extra_cmake_args=args.extra_cmake,
    )

    if not args.skip_archive:
        archive_bundle(install_dir, args.artifact_dir.resolve(), bundle_name=args.bundle_name, platform_id=platform_id)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
