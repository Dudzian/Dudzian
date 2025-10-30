"""Generator stubów gRPC dla kontraktu `proto/trading.proto`.

Skrypt pozwala wygenerować artefakty Pythonowe oraz C++ dla demona Qt/QML.
Został zaprojektowany tak, aby można było go uruchamiać lokalnie i w CI.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Tuple


def _resolve_builtin_proto_paths() -> list[str]:
    """Return include paths for well-known Google protos if available."""

    google_paths: list[str] = []
    grpc_paths: list[str] = []

    try:
        import google.protobuf  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        pass
    else:
        module_file = getattr(google.protobuf, "__file__", None)
        if module_file is not None:
            package_path = Path(module_file).resolve().parent
            google_paths.append(str(package_path.parent.parent))

    try:
        import grpc_tools  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        pass
    else:
        module_file = getattr(grpc_tools, "__file__", None)
        if module_file is not None:
            grpc_paths.append(str(Path(module_file).resolve().parent / "_proto"))

    return [*google_paths, *grpc_paths]


def _patch_python_package_imports(output_dir: Path, proto_file: str) -> None:
    """Upewnia się, że wygenerowane stuby używają względnych importów w pakiecie."""
    module_base = Path(proto_file).stem + "_pb2"
    grpc_file = output_dir / f"{module_base}_grpc.py"
    if not grpc_file.exists():  # pragma: no cover - skip gdy pominięto generowanie
        return

    text = grpc_file.read_text(encoding="utf-8")
    # Zmiana: `import <module_base> as ...` -> `from . import <module_base> as ...`
    if f"import {module_base} as" in text:
        text = text.replace(
            f"import {module_base} as",
            f"from . import {module_base} as",
        )
    # Zmiana: `import <module_base>\n` -> `from . import <module_base>\n`
    if f"import {module_base}\n" in text:
        text = text.replace(
            f"import {module_base}\n",
            f"from . import {module_base}\n",
        )
    grpc_file.write_text(text, encoding="utf-8")


def _ensure_python_package(output_dir: Path) -> None:
    """Tworzy `__init__.py`, aby katalog ze stubami był pakietem Pythona."""
    init_file = output_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# Package for generated gRPC stubs\n", encoding="utf-8")


DEFAULT_PROTO = "trading.proto"


def python_stub_targets(
    output_dir: Path | str = Path("bot_core/generated"),
    proto_file: str = DEFAULT_PROTO,
) -> Tuple[Path, ...]:
    """Zwraca oczekiwane ścieżki stubów Pythona dla danego pliku .proto."""

    directory = Path(output_dir)
    base = Path(proto_file).stem + "_pb2"
    return (
        directory / f"{base}.py",
        directory / f"{base}_grpc.py",
    )


def missing_python_stubs(
    output_dir: Path | str = Path("bot_core/generated"),
    proto_file: str = DEFAULT_PROTO,
) -> list[Path]:
    """Zwraca listę brakujących plików stubów Pythona."""

    return [path for path in python_stub_targets(output_dir, proto_file) if not path.exists()]


class StubGenerationError(RuntimeError):
    """Wyjątek zgłaszany, gdy nie możemy wygenerować stubów."""


def _run_command(cmd: list[str], dry_run: bool) -> None:
    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def _ensure_dependency(module: str, package: str) -> None:
    if module in sys.modules:
        return
    try:
        __import__(module)
    except ImportError as exc:  # pragma: no cover - defensywne logowanie
        raise StubGenerationError(
            f"Brak modułu '{module}'. Zainstaluj pakiet '{package}' (np. poetry add {package})."
        ) from exc


def _resolve_executable(name: str, override: str | None, description: str) -> str:
    if override:
        return override
    location = shutil.which(name)
    if location is None:
        raise StubGenerationError(f"Nie znaleziono {description} ('{name}') w PATH.")
    return location


def _generate_python_stubs(args: argparse.Namespace) -> None:
    if args.skip_python:
        return

    _ensure_dependency("grpc_tools", "grpcio-tools")

    output_dir = Path(args.out_python)
    output_dir.mkdir(parents=True, exist_ok=True)

    proto_file = Path(args.proto_path) / args.proto_file
    include_paths = [args.proto_path, *_resolve_builtin_proto_paths()]
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
    ]
    for path in include_paths:
        cmd.append(f"--proto_path={path}")
    cmd.extend(
        [
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            str(proto_file),
        ]
    )
    _run_command(cmd, args.dry_run)

    if not args.dry_run:
        _patch_python_package_imports(output_dir, args.proto_file)
        _ensure_python_package(output_dir)


def _validate_python_stubs(args: argparse.Namespace) -> None:
    missing = missing_python_stubs(args.out_python, args.proto_file)
    if missing:
        raise StubGenerationError(
            "Brak wygenerowanych plików stubów: " + ", ".join(str(path) for path in missing)
        )


def _generate_cpp_stubs(args: argparse.Namespace) -> None:
    if args.skip_cpp:
        return

    protoc = _resolve_executable("protoc", args.protoc, "narzędzia 'protoc'")
    plugin = _resolve_executable(
        "grpc_cpp_plugin", args.grpc_cpp_plugin, "pluginu 'grpc_cpp_plugin'"
    )

    output_dir = Path(args.out_cpp)
    output_dir.mkdir(parents=True, exist_ok=True)

    proto_file = Path(args.proto_path) / args.proto_file
    include_paths = [args.proto_path, *_resolve_builtin_proto_paths()]
    cmd = [
        protoc,
    ]
    for path in include_paths:
        cmd.append(f"--proto_path={path}")
    cmd.extend(
        [
            f"--cpp_out={output_dir}",
            f"--grpc_out={output_dir}",
            f"--plugin=protoc-gen-grpc={plugin}",
            str(proto_file),
        ]
    )
    _run_command(cmd, args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generowanie stubów trading gRPC")
    parser.add_argument(
        "--proto-path",
        default="proto",
        help="Katalog z plikami .proto (domyślnie 'proto')",
    )
    parser.add_argument(
        "--proto-file",
        default=DEFAULT_PROTO,
        help="Nazwa pliku .proto do skompilowania (domyślnie trading.proto)",
    )
    parser.add_argument(
        "--out-python",
        default="bot_core/generated",
        help="Katalog docelowy dla stubów Python (domyślnie bot_core/generated)",
    )
    parser.add_argument(
        "--out-cpp",
        default="core/generated",
        help="Katalog docelowy dla stubów C++ (domyślnie core/generated)",
    )
    parser.add_argument("--skip-python", action="store_true", help="Pomiń generowanie stubów Python")
    parser.add_argument("--skip-cpp", action="store_true", help="Pomiń generowanie stubów C++")
    parser.add_argument(
        "--protoc",
        default=None,
        help="Ścieżka do binarki protoc (gdy nie znajduje się w PATH)",
    )
    parser.add_argument(
        "--grpc-cpp-plugin",
        default=None,
        help="Ścieżka do pluginu grpc_cpp_plugin (gdy nie znajduje się w PATH)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Nie uruchamiaj protoc, jedynie wypisz komendy")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        _generate_python_stubs(args)
        if not args.dry_run and not args.skip_python:
            _validate_python_stubs(args)
        _generate_cpp_stubs(args)
    except StubGenerationError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover - obsługa CLI
    raise SystemExit(main())
