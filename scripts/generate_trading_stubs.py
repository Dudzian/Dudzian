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


def _patch_python_package_imports(output_dir: Path, proto_file: str) -> None:
    """Ensure generated stubs use relative imports inside the package."""

    module_base = Path(proto_file).stem + "_pb2"
    grpc_file = output_dir / f"{module_base}_grpc.py"
    if not grpc_file.exists():  # pragma: no cover - defensive guard for skip_python
        return

    text = grpc_file.read_text()
    if f"import {module_base} as" in text:
        text = text.replace(
            f"import {module_base} as",
            f"from . import {module_base} as",
        )
    if f"import {module_base}\n" in text:
        text = text.replace(
            f"import {module_base}\n",
            f"from . import {module_base}\n",
        )
    grpc_file.write_text(text)


DEFAULT_PROTO = "trading.proto"


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
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={args.proto_path}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        str(proto_file),
    ]
    _run_command(cmd, args.dry_run)
    if not args.dry_run:
        _patch_python_package_imports(output_dir, args.proto_file)


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
    cmd = [
        protoc,
        f"--proto_path={args.proto_path}",
        f"--cpp_out={output_dir}",
        f"--grpc_out={output_dir}",
        f"--plugin=protoc-gen-grpc={plugin}",
        str(proto_file),
    ]
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
        _generate_cpp_stubs(args)
    except StubGenerationError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover - obsługa CLI
    raise SystemExit(main())
