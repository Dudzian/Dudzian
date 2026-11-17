"""Jawna bramka diagnostycznego importu; bez site i bez zmiany sys.path.

Wymaga niezależnego sprawdzenia tego pliku, interpretera, stdlib, ELF loadera
i receipt przed startem. Nie poświadcza własnego pochodzenia ani uprawnień.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import platform
import runpy
import stat
import sysconfig


def verify_tree(root: Path, expected: dict[str, str]) -> None:
    actual: dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("SYMLINK_RUNTIME")
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode) and not (
            stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
        ):
            raise ValueError("NONREGULAR_RUNTIME")
        if path.is_file() and path != root / "ENVIRONMENT.json":
            actual[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    if actual != expected:
        raise ValueError("RUNTIME_CONTENT_MISMATCH")


class FixedRootFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root: Path) -> None:
        self.root = str(root)

    def find_spec(self, fullname: str, path: object = None, target: object = None):
        # Pakietowe __path__ jest ustawiane przez standardowy loader. Przeszukiwanie
        # top-level jest ograniczone do jednego zatwierdzonego katalogu danych.
        if path is None:
            return importlib.machinery.PathFinder.find_spec(fullname, [self.root])
        return None


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if (
        len(args) not in {3, 4}
        or args[:2] != ["--authorize-native-load", "--receipt-sha256"]
        or (len(args) == 4 and args[3] != "--authorize-readonly-preflight")
        or not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode)
    ):
        print('{"status":"REFUSED","reason":"EXPLICIT_NATIVE_LOAD_GATE_REQUIRED"}')
        return 2
    root = Path(__file__).absolute().parent
    try:
        receipt_metadata = (root / "ENVIRONMENT.json").lstat()
        if (
            not stat.S_ISREG(receipt_metadata.st_mode)
            or receipt_metadata.st_nlink != 1
            or receipt_metadata.st_size > 16 * 1024 * 1024
        ):
            raise ValueError("RECEIPT_OBJECT_REJECTED")
        raw = (root / "ENVIRONMENT.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != args[2]:
            raise ValueError("INDEPENDENT_RECEIPT_DIGEST_MISMATCH")
        receipt = json.loads(raw)
        if receipt["root"] != str(root):
            raise ValueError("RUNTIME_LOCATION_CHANGED")
        verify_tree(root, receipt["files"])
        auditor = runpy.run_path(str(root / "runtime-audit.py"))
        if (
            auditor["base_runtime_inventory"]([Path(p) for p in receipt["base_runtime"]["roots"]])
            != receipt["base_runtime"]
        ):
            raise ValueError("BASE_RUNTIME_CHANGED")
        host = receipt["host"]
        release = platform.freedesktop_os_release()
        if host != {
            "python_full_version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "system": platform.system(),
            "soabi": str(sysconfig.get_config_var("SOABI")),
            "interpreter_sha256": hashlib.sha256(
                Path(sys.executable).resolve().read_bytes()
            ).hexdigest(),
            "distribution": release.get("ID", ""),
            "distribution_version": release.get("VERSION_ID", ""),
            "glibc": str(getattr(os, "confstr")("CS_GNU_LIBC_VERSION")),
        }:
            raise ValueError("LAUNCH_HOST_MISMATCH")
        for path, digest in receipt["external_files"].items():
            candidate = Path(path)
            if (
                any(p.is_symlink() for p in (candidate, *candidate.parents))
                or hashlib.sha256(candidate.read_bytes()).hexdigest() != digest
            ):
                raise ValueError("EXTERNAL_ARTIFACT_CHANGED")
    except (ValueError, OSError, KeyError, TypeError):
        print('{"status":"REFUSED","reason":"RUNTIME_EVIDENCE_MISSING"}')
        return 2
    # -S zachowuje stdlib z zaufanej instalacji bazowej. Nie wykonuje site/.pth.
    # Finder jest po standardowych finderach: pakiet nie zastąpi stdlib.
    sys.meta_path.append(FixedRootFinder(root / "packages"))
    if len(args) == 3:
        importlib.import_module("libvirt")
        print(
            '{"status":"NATIVE_IMPORT_COMPLETED","libvirt_operations":"NOT_EXECUTED","authorization":"NOT_AUTHORIZED"}'
        )
        return 0
    from scripts.stage9_observer_kvm_preflight_v5d1 import main as preflight

    return int(preflight([]))


if __name__ == "__main__":
    raise SystemExit(main())
