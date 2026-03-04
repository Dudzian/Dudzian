"""Helpery budujące i pakujące powłokę Qt na główne platformy."""

from __future__ import annotations

import argparse
import importlib.metadata
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


def _read_qt_version(qt_prefix: Path) -> str | None:
    version_file = qt_prefix / "lib" / "cmake" / "Qt6" / "Qt6ConfigVersion.cmake"
    if not version_file.exists():
        return None

    for line in version_file.read_text().splitlines():
        if "QT_VERSION" in line:
            tokens = line.replace("(", " ").replace(")", " ").split()
            if len(tokens) >= 2 and tokens[0].startswith("set"):
                return tokens[1].strip('"')
    return None


def _major_minor(version: str | None) -> tuple[int, int] | None:
    if not version:
        return None
    parts = version.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None


def _gather_available_modules(qt_prefix: Path) -> set[str]:
    cmake_dir = qt_prefix / "lib" / "cmake"
    if not cmake_dir.is_dir():
        return set()
    return {path.name.lower() for path in cmake_dir.iterdir() if path.is_dir()}


def _norm_module(name: str) -> str:
    normalized = name.strip().lower()
    if normalized.startswith("qt6"):
        return "qt" + normalized[3:]
    return normalized


_MODULE_ALIASES: dict[str, tuple[str, ...]] = {
    "qtdeclarative": ("qtqml", "qtquick"),
}


def _looks_like_protobuf_cmake_dir(path: Path) -> bool:
    return path.is_dir() and path.name.lower() == "protobuf" and path.parent.name.lower() == "cmake"


def _probe_protobuf_dir(base: Path | str | None) -> str | None:
    if not base:
        return None
    root = Path(base)
    candidates = [
        root / "lib" / "cmake" / "protobuf",
        root / "lib" / "cmake" / "Protobuf",
        root / "lib64" / "cmake" / "protobuf",
        root / "lib64" / "cmake" / "Protobuf",
        root / "cmake" / "protobuf",
        root / "cmake" / "Protobuf",
        root / "share" / "cmake" / "protobuf",
        root / "share" / "cmake" / "Protobuf",
        root,
    ]
    lib_dir = root / "lib"
    if lib_dir.is_dir():
        for triplet_dir in lib_dir.glob("*-linux-gnu"):
            candidates.extend(
                [
                    triplet_dir / "cmake" / "protobuf",
                    triplet_dir / "cmake" / "Protobuf",
                ]
            )

    for candidate in candidates:
        if _looks_like_protobuf_cmake_dir(candidate):
            return str(candidate)
    return None


def _detect_protobuf_dir() -> str | None:
    for key in ("Protobuf_DIR", "Protobuf_ROOT"):
        env_value = os.environ.get(key)
        detected = _probe_protobuf_dir(env_value)
        if detected:
            return detected

    homebrew_prefix = os.environ.get("HOMEBREW_PREFIX")
    detected = _probe_protobuf_dir(Path(homebrew_prefix) / "opt" / "protobuf" if homebrew_prefix else None)
    if detected:
        return detected

    for prefix in ("/opt/homebrew", "/usr/local"):
        detected = _probe_protobuf_dir(Path(prefix) / "opt" / "protobuf")
        if detected:
            return detected

    # Typowe lokalizacje pakietów systemowych Linux (Debian/Ubuntu/Fedora).
    for system_path in (
        "/usr/lib/x86_64-linux-gnu/cmake/protobuf",
        "/usr/lib/x86_64-linux-gnu/cmake/Protobuf",
        "/usr/lib/aarch64-linux-gnu/cmake/protobuf",
        "/usr/lib/aarch64-linux-gnu/cmake/Protobuf",
        "/usr/lib64/cmake/protobuf",
        "/usr/lib64/cmake/Protobuf",
        "/usr/lib/cmake/protobuf",
        "/usr/lib/cmake/Protobuf",
        "/usr/local/lib/cmake/protobuf",
        "/usr/local/lib/cmake/Protobuf",
        "/usr/share/cmake/protobuf",
        "/usr/share/cmake/Protobuf",
    ):
        if Path(system_path).is_dir():
            return system_path

    brew_executable = shutil.which("brew")
    if brew_executable:
        result = subprocess.run(
            [brew_executable, "--prefix", "protobuf"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        candidate = result.stdout.strip()
        detected = _probe_protobuf_dir(Path(candidate) if candidate else None)
        if detected:
            return detected

    return None


def _protobuf_root_from_cmake_dir(protobuf_dir: str) -> str:
    path = Path(protobuf_dir)

    # Oczekiwany układ: <prefix>/lib[/<triplet>]/cmake/(protobuf|Protobuf)
    if path.name.lower() != "protobuf" or path.parent.name.lower() != "cmake":
        return str(path)

    libish = path.parent.parent  # .../lib, .../lib64 albo .../lib/<triplet>
    if libish.name.lower() in {"lib", "lib64"}:
        return str(libish.parent)

    # Debian/Ubuntu multiarch: .../lib/x86_64-linux-gnu/cmake/protobuf
    if libish.parent.name.lower() == "lib":
        return str(libish.parent.parent)

    return str(libish.parent)


def preflight(qt_prefix: str | None, required_modules: Sequence[str]) -> Path:
    detected_qt_prefix = _detect_qt_prefix(qt_prefix)
    if not detected_qt_prefix:
        raise RuntimeError("Nie wykryto instalacji Qt (brak zmiennych QT_ROOT_DIR/Qt6_DIR/Qt_DIR/QTDIR ani parametru --qt-prefix)")

    qt_prefix_path = Path(detected_qt_prefix)
    if not qt_prefix_path.exists():
        raise RuntimeError(f"Ścieżka do Qt nie istnieje: {qt_prefix_path}")

    qt_version = _read_qt_version(qt_prefix_path)
    logger.info("Wykryta instalacja Qt: %s (wersja: %s)", qt_prefix_path, qt_version or "nieznana")

    try:
        pyside_version = importlib.metadata.version("PySide6")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("PySide6 nie jest zainstalowany w środowisku") from exc

    logger.info("Wykryta wersja PySide6: %s", pyside_version)

    qt_major_minor = _major_minor(qt_version)
    pyside_major_minor = _major_minor(pyside_version)
    if qt_major_minor and pyside_major_minor and qt_major_minor != pyside_major_minor:
        raise RuntimeError(
            "Niezgodne wersje Qt/PySide6: Qt="
            f"{qt_version or 'unknown'} vs PySide6={pyside_version}. Ustaw PYSIDE6_VERSION zgodnie z instalacją Qt"
        )

    available_modules = _gather_available_modules(qt_prefix_path)
    available_modules_normalized = {_norm_module(module) for module in available_modules}
    missing_modules = []
    for module in required_modules:
        normalized = _norm_module(module)
        if not normalized:
            continue
        aliases = _MODULE_ALIASES.get(normalized)
        if aliases:
            if not any(alias in available_modules_normalized for alias in aliases):
                missing_modules.append(module)
            continue
        if normalized not in available_modules_normalized:
            missing_modules.append(module)

    if missing_modules:
        raise RuntimeError(
            "Brakuje modułów Qt w instalacji: " + ", ".join(sorted(missing_modules)) +
            f". Dostępne katalogi cmake: {', '.join(sorted(available_modules))}"
        )

    return qt_prefix_path


def _extract_cmake_define(args: Sequence[str], key: str) -> str | None:
    plain_prefix = f"-D{key}="
    typed_prefix = f"-D{key}:"
    for arg in args:
        if not (arg.startswith(plain_prefix) or arg.startswith(typed_prefix)):
            continue
        if "=" not in arg:
            continue
        return arg.split("=", 1)[1]
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

    extra_args = list(extra_cmake_args)
    protobuf_dir_override = _extract_cmake_define(extra_args, "Protobuf_DIR")
    protobuf_root_override = _extract_cmake_define(extra_args, "Protobuf_ROOT")

    if protobuf_dir_override and not protobuf_root_override:
        protobuf_root = _protobuf_root_from_cmake_dir(protobuf_dir_override)
        logger.info("Derived Protobuf root from Protobuf_DIR override: %s", protobuf_root)
        extra_args.append(f"-DProtobuf_ROOT={protobuf_root}")
    elif protobuf_root_override and not protobuf_dir_override:
        protobuf_dir = _probe_protobuf_dir(protobuf_root_override)
        if protobuf_dir:
            logger.info("Detected Protobuf CMake directory from Protobuf_ROOT override: %s", protobuf_dir)
            extra_args.append(f"-DProtobuf_DIR={protobuf_dir}")
        else:
            logger.info("Could not derive Protobuf_DIR from Protobuf_ROOT override: %s", protobuf_root_override)
    elif not protobuf_dir_override and not protobuf_root_override:
        protobuf_dir = _detect_protobuf_dir()
        if protobuf_dir:
            logger.info("Detected Protobuf CMake directory: %s", protobuf_dir)
            extra_args.append(f"-DProtobuf_DIR={protobuf_dir}")
            protobuf_root = _protobuf_root_from_cmake_dir(protobuf_dir)
            logger.info("Derived Protobuf root: %s", protobuf_root)
            extra_args.append(f"-DProtobuf_ROOT={protobuf_root}")

    cmake_args.extend(extra_args)
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

    modules_env = os.environ.get("QT_WINDOWS_MODULES" if platform_id == "windows" else "QT_DESKTOP_MODULES", "")
    required_modules = modules_env.split()

    qt_prefix_path = preflight(args.qt_prefix, required_modules)

    configure_and_build(
        source_dir,
        build_dir,
        build_type=args.build_type,
        qt_prefix=str(qt_prefix_path),
        install_prefix=install_dir,
        extra_cmake_args=args.extra_cmake,
    )

    if not args.skip_archive:
        archive_bundle(install_dir, args.artifact_dir.resolve(), bundle_name=args.bundle_name, platform_id=platform_id)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
