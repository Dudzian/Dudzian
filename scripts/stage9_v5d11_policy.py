"""Render, compile and verify the versioned AppArmor policy for a pinned A view."""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import stat
import subprocess
import sys
from typing import Any, cast

TEMPLATE = (
    Path(__file__).resolve().parents[1] / "deployment/stage9_v5d11/observer-profile-a.apparmor"
)
PARSER = Path("/usr/sbin/apparmor_parser")
SECURITYFS = Path("/sys/kernel/security/apparmor")
EXPORT_BINARY = Path("/sys/module/apparmor/parameters/export_binary")
ATOMIC_SET_LOAD = SECURITYFS / "features/policy/set_load"
LABELS = (
    "cryptohunter-observer-preflight-v5d11-a",
    "cryptohunter-observer-preflight-v5d11-qemu",
)
DIAGNOSTIC_LABELS = tuple(label + "-diagnostic" for label in LABELS)
DIAGNOSTIC_FILES = {
    "/run/stage9-v5d11/diagnostic.py",
    "/run/stage9-v5d11/diagnostic-inputs.json",
}
SOCKETS = {"/run/libvirt/libvirt-sock-ro", "/run/libvirt/virtqemud-sock-ro"}
DATA_PREFIXES = ("/etc/cryptohunter/observer-v5d1/", "/var/lib/libvirt/images/")
IMAGE_PREFIX = DATA_PREFIXES[1]
QEMU = "/usr/bin/qemu-img"
CONFIGURATION = {
    "/etc/libvirt/libvirt.conf",
    "/run/stage9-v5d11/abi-request.json",
    "/run/stage9-v5d11/launch-authorization.json",
}
PARSER_ENV = {"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"}
LIMIT = 16 * 1024 * 1024


class Refusal(ValueError):
    """A pinned policy or kernel observation failed closed."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise Refusal(reason)


def contract() -> dict[str, Any]:
    return runpy.run_path(str(Path(__file__).with_name("stage9_v5d10_contract.py")))


def literal(path: str) -> str:
    # AppArmor globs, delimiters, whitespace and variable references are excluded.
    # Dot-prefixed components are valid, while . and .. are not valid components.
    require(
        re.fullmatch(r"(?:/[A-Za-z0-9_.+@-]+)+", path) is not None
        and all(part not in {".", ".."} for part in path.split("/")[1:]),
        "APPARMOR_LITERAL_PATH_REJECTED",
    )
    return path


def read_pinned(path: Path, pin: str, schema: dict[str, Any]) -> dict[str, Any]:
    require(path.is_absolute() and schema["sha256_value"](pin), "APPARMOR_INPUT_PIN_REJECTED")
    raw = schema["secure_read"](path, limit=LIMIT)
    require(schema["digest"](raw) == pin, "APPARMOR_INPUT_PIN_MISMATCH")
    return cast(dict[str, Any], schema["loads"](raw))


def render(
    receipt: dict[str, Any],
    request: dict[str, Any],
    socket: str,
    data_paths: list[str],
    *,
    diagnostic: bool = False,
) -> bytes:
    """Build deterministic exact file rules from the already validated V5d10 inputs."""
    root = PurePosixPath(literal(receipt["root"]))
    interpreter = str(root / "venv/bin/python3.12")
    own_files = {str(root / name) for name in receipt["files"]}
    base = receipt["base_runtime"]
    external = set(receipt["external_files"]) | set(base["files"])
    all_files = own_files | external
    mapping = request["library_map"]
    extensions = request["additional_extensions"]
    libvirt = request["libvirt_extension"]
    require(
        request["profile"] == "stage9-v5d4-static-abi-v1"
        and request["interpreter"] == interpreter
        and type(mapping) is dict
        and bool(mapping)
        and all(type(name) is str and type(provider) is str for name, provider in mapping.items())
        and type(extensions) is list
        and all(type(name) is str for name in extensions)
        and type(libvirt) is str,
        "APPARMOR_ABI_SCHEMA_REJECTED",
    )
    require(
        socket in SOCKETS
        and len(data_paths) == len(set(data_paths))
        and all(name.startswith(DATA_PREFIXES) for name in data_paths),
        "APPARMOR_SCOPE_REJECTED",
    )
    system_native = set(mapping.values())
    native = system_native | set(extensions) | {libvirt, interpreter}
    require(
        system_native <= external and native <= all_files and {interpreter, QEMU} <= native,
        "APPARMOR_NATIVE_PIN_REJECTED",
    )
    require(
        all(name.startswith("/usr/lib/") for name in system_native),
        "APPARMOR_LIBRARY_SCOPE_REJECTED",
    )

    # Include these precise lookup spellings too. The kernel normally mediates
    # resolved provider paths, including exec through a procfs file descriptor.
    alias_rules: dict[str, str] = {}
    for name, provider in mapping.items():
        if name == "/lib64/ld-linux-x86-64.so.2":
            alias_rules[name] = provider
            alias_rules["/usr" + name] = provider
            continue
        require("/" not in name, "APPARMOR_LIBRARY_NAME_REJECTED")
        for prefix in ("/lib/x86_64-linux-gnu/", "/usr/lib/x86_64-linux-gnu/"):
            alias_rules[prefix + name] = provider
    for name, row in base["symlinks"].items():
        require(
            name not in alias_rules or alias_rules[name] == row["resolved"],
            "APPARMOR_ALIAS_CONFLICT",
        )
        alias_rules[name] = row["resolved"]
    require(
        all(name not in all_files or name == provider for name, provider in alias_rules.items()),
        "APPARMOR_ALIAS_CONFLICT",
    )

    images = {name for name in data_paths if name.startswith(IMAGE_PREFIX)}
    observer = {name: "mr" if name in native else "r" for name in all_files - {interpreter, QEMU}}
    observer.update(
        {name: "mr" if provider in native else "r" for name, provider in alias_rules.items()}
    )
    observer.update({name: "rk" if name in images else "r" for name in data_paths})
    observer.update(dict.fromkeys(CONFIGURATION | {str(root / "ENVIRONMENT.json")}, "r"))
    if diagnostic:
        observer = {
            path: mode
            for path, mode in observer.items()
            if not path.startswith((f"{root}/packages/scripts/", f"{root}/packages/deployment/"))
            and path not in {f"{root}/launch.py", f"{root}/contract.py", f"{root}/runtime-audit.py"}
        }
        observer.update(dict.fromkeys(DIAGNOSTIC_FILES, "r"))
    observer[socket] = "rw"
    observer[interpreter] = "mrix"
    qemu = dict.fromkeys(
        system_native
        | {name for name, provider in alias_rules.items() if provider in system_native},
        "mr",
    )
    qemu.update(dict.fromkeys(images, "rk"))
    own_dirs = {str(root / name) for name in receipt["directories"]}
    observer_source = policy_rules(observer, own_dirs | {"/proc", "/dev"})
    qemu_source = policy_rules(qemu, {"/dev", "/proc"})
    template = TEMPLATE.read_text(encoding="utf-8")
    require(
        template.count("@OBSERVER_RULES@") == template.count("@QEMU_RULES@") == 1,
        "APPARMOR_TEMPLATE_REJECTED",
    )
    if diagnostic:
        require(
            template.count("rmPx") == 1 and template.count("deny ptrace,") == 2,
            "APPARMOR_DIAGNOSTIC_TEMPLATE_REJECTED",
        )
        for label in LABELS:
            template = template.replace(label, label + "-diagnostic")
        # Px clears LD_DEBUG. The diagnostic entrypoint supplies the complete fixed env.
        # Host procfs reads require readby even when the root reader has CAP_SYS_PTRACE.
        template = template.replace("rmPx", "rmpx").replace(
            "deny ptrace,",
            "ptrace (readby) peer=unconfined,\n  deny ptrace (read, trace, tracedby),",
        )
    return (
        template.replace("@OBSERVER_RULES@", observer_source)
        .replace("@QEMU_RULES@", qemu_source)
        .encode("utf-8")
    )


def policy_rules(files: dict[str, str], directories: set[str]) -> str:
    for name in files:
        directories.update(str(parent) for parent in PurePosixPath(literal(name)).parents)
    directory_rules = {"/": "r"}
    for name in directories - {"/"}:
        directory_rules[literal(name) + "/"] = "r"
    return "\n  ".join(
        f'"{path}" {mode},' for path, mode in sorted({**files, **directory_rules}.items())
    )


def parser_run(arguments: list[str], source: bytes) -> bytes:
    require(sys.platform == "linux" and PARSER.is_file(), "APPARMOR_PARSER_UNAVAILABLE")
    result = subprocess.run(
        [str(PARSER), "--config-file", "/dev/null", *arguments],
        input=source,
        capture_output=True,
        cwd="/",
        env=PARSER_ENV,
        timeout=60,
        check=False,
    )
    require(result.returncode == 0, "APPARMOR_PARSER_REFUSED")
    return result.stdout


def compile_policy(source: bytes) -> bytes:
    raw = parser_run(
        [
            "--warn=rule-not-enforced",
            "--warn=rule-downgraded",
            "--warn=abi",
            "--Werror",
            "--jobs",
            "1",
            "--skip-cache",
            "--skip-kernel-load",
            "--stdout",
        ],
        source,
    )
    require(0 < len(raw) <= LIMIT, "APPARMOR_COMPILE_OUTPUT_REJECTED")
    return raw


def fs_read(path: Path, max_bytes: int = LIMIT) -> bytes:
    # Securityfs seq_files can report st_size=0 even when their content is present.
    with path.open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    require(0 < len(raw) <= max_bytes, "APPARMOR_KERNEL_EVIDENCE_REJECTED")
    return raw


def kernel_revision(root: Path) -> str:
    # A second read on this kernel descriptor waits for the next revision.
    descriptor = os.open(root / "revision", os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        raw = os.read(descriptor, 32)
    finally:
        os.close(descriptor)
    require(re.fullmatch(rb"[0-9]+\n", raw) is not None, "APPARMOR_REVISION_REJECTED")
    return raw.removesuffix(b"\n").decode("ascii")


def active_profiles(
    expected: bytes, root: Path = SECURITYFS, *, labels: tuple[str, ...] = LABELS
) -> tuple[str, dict[str, str]]:
    """Verify exported kernel blob for both exact enforced labels in one revision."""
    before = kernel_revision(root)
    entries = root / "policy/profiles"
    active: dict[str, str] = {}
    for entry in entries.iterdir():
        if not any(re.fullmatch(re.escape(name) + r"\.[0-9]+", entry.name) for name in labels):
            continue
        require(stat.S_ISDIR(entry.lstat().st_mode), "APPARMOR_PROFILE_DIRECTORY_REJECTED")
        name = fs_read(entry / "name", 128).removesuffix(b"\n").decode("ascii")
        require(
            name in labels and entry.name.startswith(name + ".") and name not in active,
            "APPARMOR_PROFILE_ID_REJECTED",
        )
        require(fs_read(entry / "mode", 32) == b"enforce\n", "APPARMOR_PROFILE_MODE_REJECTED")
        raw = entry / "raw_data"
        require(
            re.fullmatch(r"\.\./\.\./raw_data/[0-9]+/raw_data", os.readlink(raw)) is not None,
            "APPARMOR_RAW_EXPORT_LINK_REJECTED",
        )
        require(fs_read(raw) == expected, "APPARMOR_KERNEL_POLICY_MISMATCH")
        active[name] = "enforce"
    require(set(active) == set(labels), "APPARMOR_PROFILE_SET_REJECTED")
    require(kernel_revision(root) == before, "APPARMOR_POLICY_CHANGED_DURING_READ")
    return before, active


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("render", "compile", "load", "check"))
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--abi-request", required=True, type=Path)
    parser.add_argument("--socket", required=True, choices=sorted(SOCKETS))
    parser.add_argument("--data-path", action="append", default=[])
    parser.add_argument("--source-sha256")
    parser.add_argument("--compiled-sha256")
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args(argv)
    try:
        schema = contract()
        receipt = read_pinned(args.receipt, args.receipt_sha256, schema)
        schema["validate_receipt"](receipt)
        request = read_pinned(args.abi_request, receipt["abi_request_sha256"], schema)
        source = render(receipt, request, args.socket, args.data_path, diagnostic=args.diagnostic)
        if args.mode in {"render", "compile"}:
            require(
                args.source_sha256 is None and args.compiled_sha256 is None,
                "APPARMOR_UNEXPECTED_POLICY_PINS",
            )
            sys.stdout.buffer.write(source if args.mode == "render" else compile_policy(source))
            return 0
        require(schema["digest"](source) == args.source_sha256, "APPARMOR_SOURCE_PIN_MISMATCH")
        binary = compile_policy(source)
        require(schema["digest"](binary) == args.compiled_sha256, "APPARMOR_COMPILED_PIN_MISMATCH")
        require(fs_read(EXPORT_BINARY, 32) == b"Y\n", "APPARMOR_KERNEL_EXPORT_DISABLED")
        if args.mode == "load":
            require(fs_read(ATOMIC_SET_LOAD, 32) == b"yes\n", "APPARMOR_ATOMIC_LOAD_UNAVAILABLE")
            parser_run(["--binary", "--add"], binary)
        revision, profiles = (
            active_profiles(binary, labels=DIAGNOSTIC_LABELS)
            if args.diagnostic
            else active_profiles(binary)
        )
        evidence = {
            "profile": "stage9-v5d11-apparmor-verification-v1",
            "source_sha256": args.source_sha256,
            "compiled_sha256": args.compiled_sha256,
            "exported_sha256": schema["digest"](binary),
            "kernel_revision": revision,
            "profiles": profiles,
        }
        sys.stdout.buffer.write(schema["canonical"](evidence) + b"\n")
        return 0
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        reason = str(exc) if isinstance(exc, Refusal) else "APPARMOR_POLICY_REJECTED"
        print(f"REFUSED {reason}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
