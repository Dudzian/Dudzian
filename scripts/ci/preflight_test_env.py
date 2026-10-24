from __future__ import annotations

import argparse
import importlib
import subprocess
import sys


REQUIRED_MODULES = {
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "yaml": "PyYAML",
    "numpy": "numpy",
    "joblib": "joblib",
    "pytest": "pytest",
    "pytest_cov": "pytest-cov",
    "httpx": "httpx",
}

WINDOWS_REQUIRED_MODULES = {
    "ntsecuritycon": "pywin32 (ntsecuritycon)",
    "win32security": "pywin32 (win32security)",
    "win32serviceutil": "pywin32 (win32serviceutil)",
}


def _import_module(name: str):
    try:
        return importlib.import_module(name)
    except (ModuleNotFoundError, ImportError) as exc:
        raise RuntimeError(name) from exc


def _required_modules(platform: str, *, require_fastapi: bool = False) -> dict[str, str]:
    required_modules = dict(REQUIRED_MODULES)
    if require_fastapi:
        required_modules["fastapi"] = "fastapi"
    if platform == "win32":
        required_modules.update(WINDOWS_REQUIRED_MODULES)
    return required_modules


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate CI test dependencies")
    parser.add_argument(
        "--require-fastapi", action="store_true", help="Require the endpoint test stack"
    )
    args = parser.parse_args(argv)
    missing: list[str] = []
    versions: dict[str, str] = {}
    pandas_available = False

    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    pip_version = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if pip_version.stdout:
        print(f"pip: {pip_version.stdout.strip()}")
    if pip_version.stderr:
        print(f"pip stderr: {pip_version.stderr.strip()}")
    if pip_version.returncode != 0:
        print(f"pip returncode: {pip_version.returncode}")

    for module_name, display_name in _required_modules(
        sys.platform, require_fastapi=args.require_fastapi
    ).items():
        try:
            module = _import_module(module_name)
        except RuntimeError as exc:
            missing.append(display_name)
            continue
        version = getattr(module, "__version__", "unknown")
        versions[display_name] = version
        if module_name == "pandas":
            pandas_available = True

    if pandas_available:
        pandas_show = subprocess.run(
            [sys.executable, "-m", "pip", "show", "pandas"],
            capture_output=True,
            text=True,
            check=False,
        )
        if pandas_show.stdout:
            print(pandas_show.stdout.strip())
        if pandas_show.stderr:
            print(pandas_show.stderr.strip())
        if pandas_show.returncode != 0:
            print(f"pip show pandas returncode: {pandas_show.returncode}")

    if missing:
        print("Brakuje zależności środowiska CI:")
        for name in missing:
            print(f"- {name}")
        print('Zainstaluj je poleceniem: python -m pip install -e ".[test]"')
        return 1

    print("Preflight OK:")
    for name, version in sorted(versions.items()):
        print(f"- {name}: {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
