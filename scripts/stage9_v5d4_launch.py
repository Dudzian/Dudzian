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
from typing import Any

V5D7_RECEIPT = "stage9-v5d7-prepared-environment-v1"
V5D7_VERIFICATION = "stage9-v5d7-dependency-verification-v1"


def verify_tree(
    root: Path, expected: dict[str, str], expected_directories: list[str] | None = None
) -> None:
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("SYMLINK_RUNTIME")
    actual: dict[str, str] = {}
    directories: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("SYMLINK_RUNTIME")
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode) and not (
            stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
        ):
            raise ValueError("NONREGULAR_RUNTIME")
        if stat.S_ISDIR(metadata.st_mode):
            directories.append(path.relative_to(root).as_posix())
        if path.is_file() and path != root / "ENVIRONMENT.json":
            actual[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    if actual != expected:
        raise ValueError("RUNTIME_CONTENT_MISMATCH")
    if expected_directories is not None and sorted(directories) != expected_directories:
        raise ValueError("RUNTIME_DIRECTORY_MISMATCH")


def verify_shared_receipt(root: Path, receipt: dict[str, Any]) -> None:
    """Recheck exact owners, independent expectations and installed bytes.

    The external receipt digest and explicit native gate are still mandatory;
    this consistency check cannot authenticate or authorize a local receipt.
    """
    if type(receipt) is not dict or type(receipt.get("files")) is not dict:
        raise ValueError("SHARED_WHEEL_RECEIPT_SCHEMA")
    fields = {
        "wheel_materialization",
        "shared_wheel_members",
        "shared_wheel_expectations",
        "shared_wheel_policy",
        "shared_wheel_output",
    }
    profile = receipt.get("profile")
    v5d7 = profile == V5D7_RECEIPT
    v5d5 = profile == "stage9-v5d5-prepared-environment-v1"
    v5d7_audit = "shared-audit-v5d7.py"
    if (
        profile == "stage9-v5d4-prepared-environment-v1"
        and not fields.intersection(receipt)
        and "shared-audit.py" not in receipt["files"]
        and v5d7_audit not in receipt["files"]
    ):
        return
    if (
        not (v5d7 or v5d5)
        or not fields <= set(receipt)
        or "directories" not in receipt
        or "shared-audit.py" not in receipt["files"]
        or (v5d7_audit in receipt["files"]) != v5d7
    ):
        raise ValueError("SHARED_WHEEL_RECEIPT_INCOMPLETE")
    tools = runpy.run_path(str(root / (v5d7_audit if v5d7 else "shared-audit.py")))
    packages = receipt["wheel_materialization"]
    shared = tools["verify_shared"](packages, receipt["shared_wheel_expectations"])
    if receipt["shared_wheel_policy"] != tools["POLICY_ID"] or not tools["exact"](
        receipt["shared_wheel_members"], shared
    ):
        raise ValueError("SHARED_WHEEL_RECEIPT_MISMATCH")
    if v5d7:
        verification = receipt.get("dependency_verification")
        if (
            type(verification) is not dict
            or verification.get("profile") != V5D7_VERIFICATION
            or any(
                key not in verification or not tools["exact"](verification[key], receipt[key])
                for key in (
                    "shared_wheel_policy",
                    "shared_wheel_members",
                    "shared_wheel_expectations",
                )
            )
        ):
            raise ValueError("SHARED_WHEEL_VERIFICATION_MISMATCH")
    if (
        type(receipt.get("package_files")) is not dict
        or type(receipt.get("wheel_archives")) is not list
        or not all(
            type(package) is dict
            and all(
                type(package.get(key)) is str for key in ("name", "version", "filename", "sha256")
            )
            for package in receipt["wheel_archives"]
        )
    ):
        raise ValueError("SHARED_WHEEL_RECEIPT_SCHEMA")
    identities = [
        {key: package[key] for key in ("name", "version", "filename", "sha256")}
        for package in packages
    ]
    archives = [
        {key: package[key] for key in ("name", "version", "filename", "sha256")}
        for package in receipt["wheel_archives"]
    ]
    if sorted(identities, key=lambda row: row["name"]) != sorted(
        archives, key=lambda row: row["name"]
    ):
        raise ValueError("SHARED_WHEEL_ARCHIVE_INVENTORY_MISMATCH")
    for package in packages:
        for member in package["materialized_members"].values():
            name, digest = member["path"], member["sha256"]
            if (
                receipt["package_files"].get(name) != digest
                or receipt["files"].get("packages/" + name) != digest
            ):
                raise ValueError("SHARED_WHEEL_MAPPING_MISMATCH")
    expected_output = [
        {"path": row["path"], "sha256": row["sha256"], "mode": 0o600, "type": "regular"}
        for row in shared["members"]
    ]
    if not tools["exact"](receipt["shared_wheel_output"], expected_output):
        raise ValueError("SHARED_WHEEL_OUTPUT_MISMATCH")
    for row in expected_output:
        path = root / "packages" / row["path"]
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != row["mode"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]
        ):
            raise ValueError("SHARED_WHEEL_OUTPUT_CHANGED")


def verify_header_receipt(root: Path, receipt: dict[str, Any]) -> None:
    """Bind the separate omission approval and prove absence from runtime."""
    if type(receipt) is not dict or type(receipt.get("files")) is not dict:
        raise ValueError("HEADER_OMISSION_RECEIPT_SCHEMA")
    fields = {
        "header_omissions",
        "header_omission_expectations",
        "header_omission_policy",
        "header_runtime_inclusion",
    }
    archives = receipt.get("wheel_archives", [])
    packages = receipt.get("wheel_materialization", [])
    candidate = any(
        type(package) is dict
        and (
            package.get("header_omissions")
            or (package.get("name") == "greenlet" and package.get("version") == "3.5.6")
        )
        for package in (
            (archives if type(archives) is list else [])
            + (packages if type(packages) is list else [])
        )
    )
    if (
        not candidate
        and not fields.intersection(receipt)
        and "header-audit.py" not in receipt["files"]
    ):
        return
    if (
        not fields <= set(receipt)
        or "header-audit.py" not in receipt["files"]
        or type(packages) is not list
        or type(archives) is not list
        or type(receipt.get("package_files")) is not dict
    ):
        raise ValueError("HEADER_OMISSION_RECEIPT_INCOMPLETE")
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("SYMLINK_RUNTIME")
    tools = runpy.run_path(str(root / "header-audit.py"))
    inventory = tools["verify_headers"](packages, receipt["header_omission_expectations"])
    if (
        receipt["header_omission_policy"] != tools["POLICY_ID"]
        or receipt["header_runtime_inclusion"] != "EXCLUDED"
        or not tools["exact"](receipt["header_omissions"], inventory)
    ):
        raise ValueError("HEADER_OMISSION_RECEIPT_MISMATCH")
    verification = receipt.get("dependency_verification")
    if (
        type(verification) is not dict
        or not tools["exact"](verification.get("header_omissions"), inventory)
        or not tools["exact"](
            verification.get("header_expectations"), receipt["header_omission_expectations"]
        )
        or verification.get("header_policy") != tools["POLICY_ID"]
    ):
        raise ValueError("HEADER_OMISSION_VERIFICATION_MISMATCH")
    for row in inventory["members"]:
        identity = {
            "name": row["distribution"],
            "version": row["version"],
            "filename": row["wheel_filename"],
            "sha256": row["wheel_sha256"],
        }
        if (
            sum(
                type(package) is dict
                and all(package.get(key) == value for key, value in identity.items())
                for package in archives
            )
            != 1
        ):
            raise ValueError("HEADER_OMISSION_ARCHIVE_IDENTITY_MISMATCH")
        forbidden = {row["member"], Path(row["member"]).name}
        if any(
            name in forbidden or digest == row["sha256"]
            for name, digest in receipt["package_files"].items()
        ) or any(
            name.removeprefix("packages/") in forbidden or digest == row["sha256"]
            for name, digest in receipt["files"].items()
        ):
            raise ValueError("HEADER_OMISSION_MATERIALIZED")
        # A changed receipt must not hide a copied header under an alias or a
        # symlink. The complete tree check remains mandatory in main().
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("SYMLINK_RUNTIME")
            if path.is_file() and (
                path.relative_to(root).as_posix().removeprefix("packages/") in forbidden
                or hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
            ):
                raise ValueError("HEADER_OMISSION_MATERIALIZED")


def verify_artifact_receipt(root: Path, receipt: dict[str, Any]) -> None:
    """Recheck V5d7 omission approval and absence using the verified tree inventory."""
    fields = {"artifact_omissions", "artifact_expectations", "artifact_policy"}
    files = receipt["files"]
    packages = receipt.get("wheel_materialization")
    verification = receipt.get("dependency_verification")
    v5d7 = receipt.get("profile") == V5D7_RECEIPT
    if not v5d7:
        if (
            fields.intersection(receipt)
            or "artifact-audit.py" in files
            or (type(verification) is dict and verification.get("profile") == V5D7_VERIFICATION)
            or (
                type(packages) is list
                and any(type(item) is dict and "artifact_omissions" in item for item in packages)
            )
        ):
            raise ValueError("ARTIFACT_OMISSION_RECEIPT_PROFILE")
        return
    if (
        not fields <= set(receipt)
        or "artifact-audit.py" not in files
        or type(packages) is not list
        or not all(type(item) is dict and "artifact_omissions" in item for item in packages)
        or type(receipt.get("package_files")) is not dict
        or type(verification) is not dict
        or verification.get("profile") != V5D7_VERIFICATION
    ):
        raise ValueError("ARTIFACT_OMISSION_RECEIPT_INCOMPLETE")
    tools = runpy.run_path(str(root / "artifact-audit.py"))
    inventory = tools["verify_omissions"](packages, receipt["artifact_expectations"])
    if (
        receipt["artifact_policy"] != tools["POLICY_ID"]
        or not tools["exact"](receipt["artifact_omissions"], inventory)
        or any(
            key not in verification or not tools["exact"](verification[key], receipt[key])
            for key in fields
        )
    ):
        raise ValueError("ARTIFACT_OMISSION_RECEIPT_MISMATCH")
    tools["verify_runtime_absence"](files, inventory)
    tools["verify_runtime_absence"](receipt["package_files"], inventory)


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
    if sys.platform != "linux":
        print('{"status":"REFUSED","reason":"UBUNTU_LINUX_HOST_REQUIRED"}')
        return 2
    if not (
        callable(getattr(platform, "freedesktop_os_release", None))
        and callable(getattr(os, "confstr", None))
    ):
        print('{"status":"REFUSED","reason":"LINUX_HOST_CAPABILITY_MISSING"}')
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
        if type(receipt) is not dict:
            raise ValueError("RECEIPT_SCHEMA_REJECTED")
        if receipt["root"] != str(root):
            raise ValueError("RUNTIME_LOCATION_CHANGED")
        verify_tree(root, receipt["files"], receipt.get("directories"))
        verify_shared_receipt(root, receipt)
        verify_header_receipt(root, receipt)
        verify_artifact_receipt(root, receipt)
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
