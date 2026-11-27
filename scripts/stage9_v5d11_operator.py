"""Prepare a private Profile A view and verify review for its live measurement."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import runpy
import secrets
import select
import stat
import subprocess
import sys
import uuid
import tempfile
import time
from typing import IO, Any, cast

MANIFEST = "stage9-v5d11-operator-inputs-v1"
REVIEW = "stage9-v5d11-isolation-review-v1"
PROTOCOL = "stage9-v5d11-bootstrap-v1"
EVIDENCE = "stage9-v5d11-isolation-measurement-v1"
AUTHORITY = "/run/stage9-v5d11/launch-authorization.json"
ABI_REQUEST = "/run/stage9-v5d11/abi-request.json"
CLIENT_CONFIG = "/etc/libvirt/libvirt.conf"
CLIENT_BYTES = b'uri_default = "qemu:///system"\n'
APPARMOR_LABEL = "cryptohunter-observer-preflight-v5d11-a (enforce)"
COMMAND_ENV = {"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"}
LIMIT = 1024 * 1024
CONTROL_TIMEOUT = 3600
DIAGNOSTIC_SPEC = "stage9-v5d11-diagnostic-inputs-v1"
DIAGNOSTIC_REVIEW = "stage9-v5d11-diagnostic-authorization-v1"
DIAGNOSTIC_SCRIPT = "/run/stage9-v5d11/diagnostic.py"
DIAGNOSTIC_CONFIG = "/run/stage9-v5d11/diagnostic-inputs.json"
ED25519_SPKI_PREFIX = bytes.fromhex("302a300506032b6570032100")
DATA_PREFIXES = ("/etc/cryptohunter/observer-v5d1/", "/var/lib/libvirt/images/")


class Refusal(ValueError):
    """An operator input, view or independent review was refused."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise Refusal(reason)


def tool(name: str) -> dict[str, Any]:
    return runpy.run_path(str(Path(__file__).with_name(f"stage9_{name}.py")))


def protected(path: Path, *, private: bool = False, directory: bool = False) -> None:
    """For privileged inputs, require real protected paths from / to the leaf."""
    require(path.is_absolute(), "ABSOLUTE_OPERATOR_PATH_REQUIRED")
    for item in reversed((path, *path.parents)):
        info = item.lstat()
        is_dir = item != path or directory
        require(
            info.st_uid == 0
            and info.st_mode & 0o022 == 0
            and (stat.S_ISDIR(info.st_mode) if is_dir else stat.S_ISREG(info.st_mode))
            and (is_dir or info.st_nlink == 1),
            "UNPROTECTED_OPERATOR_PATH",
        )
    if private:
        require(path.stat().st_mode & 0o077 == 0, "OPERATOR_PATH_NOT_PRIVATE")


def pinned(
    path: str | Path, expected: str, contract: dict[str, Any], *, limit: int = LIMIT
) -> bytes:
    require(contract["path_value"](str(path), absolute=True), "INPUT_PATH_REJECTED")
    raw = cast(bytes, contract["secure_read"](Path(path), limit=limit))
    require(contract["digest"](raw) == expected, "INPUT_PIN_MISMATCH")
    return raw


def read_manifest(raw: bytes, contract: dict[str, Any], launch: dict[str, Any]) -> dict[str, Any]:
    doc: dict[str, Any] = contract["loads"](raw)
    paths = {"root", "approval", "abi_request", "bootstrap", "review_key"}
    digests = {
        "receipt_sha256",
        "approval_sha256",
        "bootstrap_sha256",
        "review_key_sha256",
        "apparmor_source_sha256",
        "apparmor_compiled_sha256",
        "seccomp_sha256",
    }
    required = {"profile", "uid", "gid", "libvirt_socket", "data_files", *paths, *digests}
    require(
        required <= set(doc) <= required | {"scratch"}
        and doc["profile"] == MANIFEST
        and all(contract["path_value"](doc[key], absolute=True) for key in paths)
        and all(contract["sha256_value"](doc[key]) for key in digests)
        and all(type(doc[key]) is int and 0 < doc[key] < 2**32 - 1 for key in ("uid", "gid"))
        and doc["libvirt_socket"] in launch["LOCAL_SOCKETS"],
        "OPERATOR_MANIFEST_REJECTED",
    )
    require(type(doc["data_files"]) is dict, "OPERATOR_DATA_REJECTED")
    for destination, row in doc["data_files"].items():
        require(
            contract["path_value"](destination, absolute=True)
            and destination.startswith(DATA_PREFIXES)
            and type(row) is dict
            and set(row) == {"source", "sha256"}
            and contract["path_value"](row["source"], absolute=True)
            and contract["sha256_value"](row["sha256"]),
            "OPERATOR_DATA_REJECTED",
        )
    require(
        ("scratch" not in doc or contract["path_value"](doc["scratch"], absolute=True))
        and (
            not any(name.startswith(DATA_PREFIXES[1]) for name in doc["data_files"])
            or "scratch" in doc
        ),
        "IMAGE_SCRATCH_REQUIRED",
    )
    return doc


def write(path: Path, data: bytes, mode: int) -> None:
    set_mode = getattr(os, "fchmod", None)
    require(callable(set_mode), "POSIX_FILE_MODE_REQUIRED")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, "wb") as stream:
        assert set_mode is not None
        set_mode(stream.fileno(), mode)
        stream.write(data)


def validate_aliases(aliases: dict[str, str], targets: dict[str, str], files: set[str]) -> None:
    """Resolve literal aliases as the kernel would inside /, without host fallback."""
    require(set(aliases) == set(targets) and not set(aliases) & files, "ABI_ALIAS_SET_REJECTED")
    for alias, expected in targets.items():
        resolved: list[str] = []
        pending = list(Path(alias).parts[1:])
        expansions = 0
        while pending:
            part = pending.pop(0)
            if part == "..":
                if resolved:
                    resolved.pop()
                continue
            resolved.append(part)
            current = "/" + "/".join(resolved)
            if current in aliases:
                expansions += 1
                require(expansions <= 40, "ABI_ALIAS_LOOP")
                link = Path(aliases[current])
                if link.is_absolute():
                    resolved.clear()
                    pending[:0] = link.parts[1:]
                else:
                    resolved.pop()
                    pending[:0] = link.parts
        require("/" + "/".join(resolved) == expected, "ABI_ALIAS_TARGET_REJECTED")
        require(
            expected in files
            or any(path.startswith(expected + "/") for path in files | aliases.keys()),
            "ABI_ALIAS_PROVIDER_MISSING",
        )


def stage(
    tree: Path,
    manifest: dict[str, Any],
    contract: dict[str, Any],
    launch: dict[str, Any],
    *,
    images: Any = None,
) -> tuple[dict[str, str], tuple[int, ...], dict[str, Any]]:
    """Copy only independently pinned bytes. The physical A tree is opened read-only."""
    root = Path(manifest["root"])
    receipt_pin = manifest["receipt_sha256"]
    approval_pin = manifest["approval_sha256"]
    approval = contract["loads"](pinned(manifest["approval"], approval_pin, contract))
    contract["validate_approval"](approval)
    receipt, _ = launch["verify"](root, receipt_pin, approval_pin)
    for field in contract["APPROVAL_DIGESTS"]:
        require(receipt[field] == approval[field], "RECEIPT_APPROVAL_BINDING_REJECTED")
    require(
        receipt["host"] == approval["host"]
        and receipt["base_runtime"] == approval["base_runtime"]
        and all(
            receipt["external_files"].get(name) == sha
            for name, sha in approval["host_tools"].items()
        ),
        "RECEIPT_APPROVAL_BINDING_REJECTED",
    )
    request_bytes = pinned(manifest["abi_request"], receipt["abi_request_sha256"], contract)
    request = contract["loads"](request_bytes)
    base = receipt["base_runtime"]
    require(not base["excluded"], "BASE_RUNTIME_EXCLUSIONS_REJECTED")
    uid, gid = manifest["uid"], manifest["gid"]
    own_dirs = {str(root / name): mode for name, mode in receipt["directories"].items()}
    own_files = {str(root / name): row for name, row in receipt["files"].items()}
    receipt_path = str(root / launch["RECEIPT"])
    own_files[receipt_path] = {"sha256": receipt_pin, "mode": 0o600}
    require(
        own_files[str(root / "venv/bin/python3.12")]["sha256"]
        == receipt["host"]["interpreter_sha256"],
        "INTERPRETER_PIN_REJECTED",
    )
    system = {**base["files"], **receipt["external_files"]}
    require(
        all(system[name] == value for name, value in base["files"].items()),
        "CONFLICTING_RUNTIME_PIN",
    )
    socket = manifest["libvirt_socket"]
    special = {socket, "/dev/null", AUTHORITY, ABI_REQUEST, CLIENT_CONFIG}
    extra = manifest["data_files"]
    require(
        len(set(own_files) | set(system) | special | set(extra))
        == len(own_files) + len(system) + len(special) + len(extra),
        "VIEW_ENTRY_COLLISION",
    )
    targets = launch["_abi_link_targets"](request, base["symlinks"])
    aliases = {**targets, **{name: row["target"] for name, row in base["symlinks"].items()}}
    source_paths = {*own_files, *system, *extra}
    all_files = source_paths | special
    validate_aliases(aliases, targets, source_paths)
    forbidden = {
        "/etc/ld.so.cache",
        "/etc/ld.so.preload",
        "/usr/lib/python312.zip",
        "/usr/lib/x86_64-linux-gnu/qemu",
        "/usr/lib/x86_64-linux-gnu/libvirt/connection-driver",
    }
    dirs = {str(parent) for name in all_files | set(aliases) for parent in Path(name).parents}
    dirs.update(own_dirs)
    dirs.add("/proc")
    require(
        not (dirs & (all_files | set(aliases)))
        and not (all_files | set(aliases) | dirs) & forbidden
        and not any(name.startswith("/proc/") for name in all_files | set(aliases) | dirs),
        "VIEW_LAYOUT_REJECTED",
    )
    root_paths = {
        name for name in all_files | set(aliases) | dirs if Path(name).is_relative_to(root)
    }
    require(root_paths == set(own_files) | set(own_dirs), "PROFILE_PATH_COLLISION")
    set_owner = getattr(os, "chown", None)
    require(callable(set_owner), "POSIX_FILE_OWNER_REQUIRED")
    assert set_owner is not None
    for name in sorted(dirs - {"/"}, key=lambda item: (item.count("/"), item)):
        dest = tree / name[1:]
        dest.mkdir(mode=0o700)
        dest.chmod(own_dirs.get(name, 0o755))
        if name in own_dirs:
            set_owner(dest, uid, gid)
    for name, row in own_files.items():
        dest = tree / name[1:]
        write(dest, pinned(name, row["sha256"], contract, limit=contract["LIMIT"]), row["mode"])
        set_owner(dest, uid, gid)
    for name, sha in system.items():
        write(
            tree / name[1:],
            pinned(name, sha, contract, limit=contract["LIMIT"]),
            0o755 if name == contract["QEMU_IMG"] else 0o644,
        )
    for name, row in extra.items():
        if name.startswith(DATA_PREFIXES[1]):
            require(images is not None, "IMMUTABLE_IMAGE_STAGE_REQUIRED")
            images.copy(tree, name, Path(row["source"]), row["sha256"], gid)
            continue
        raw = pinned(row["source"], row["sha256"], contract, limit=contract["LIMIT"])
        require(not raw.startswith((b"\x7fELF", b"#!", b"PK\x03\x04")), "EXECUTABLE_DATA_REJECTED")
        write(tree / name[1:], raw, 0o444)
    for name, target in aliases.items():
        (tree / name[1:]).symlink_to(target)
    for name, raw in (
        (AUTHORITY, b""),
        (ABI_REQUEST, request_bytes),
        (CLIENT_CONFIG, CLIENT_BYTES),
        (socket, b""),
        ("/dev/null", b""),
    ):
        write(tree / name[1:], raw, 0o444)
    info = Path(socket).lstat()
    require(stat.S_ISSOCK(info.st_mode) and info.st_uid == 0, "SOURCE_SOCKET_REJECTED")
    return (
        aliases,
        (info.st_dev, info.st_ino, info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)),
        receipt,
    )


def read_frame(
    stream: IO[bytes],
    socket: str,
    contract: dict[str, Any],
    launch: dict[str, Any],
    *,
    diagnostic: bool = False,
    images: bool = False,
) -> tuple[dict[str, Any], bytes]:
    header: dict[str, Any] = contract["loads"](stream.readline(8192))
    fields = {"protocol", "mountinfo_bytes", "namespaces", "apparmor_label", "socket_identity"}
    if diagnostic:
        fields.add("host_pid")
    if images:
        fields.add("images_verified")
    label = APPARMOR_LABEL.replace(" (", "-diagnostic (") if diagnostic else APPARMOR_LABEL
    require(
        set(header) == fields
        and header["protocol"] == PROTOCOL
        and header["apparmor_label"] == label
        and (not diagnostic or type(header["host_pid"]) is int and header["host_pid"] > 1)
        and (not images or header["images_verified"] is True)
        and type(header["mountinfo_bytes"]) is int
        and 0 < header["mountinfo_bytes"] <= 4 * LIMIT,
        "BOOTSTRAP_FRAME_REJECTED",
    )
    namespaces = header["namespaces"]
    identity = header["socket_identity"]
    require(
        type(namespaces) is dict
        and set(namespaces) == {"mnt", "pid", "net"}
        and all(type(i) is int and i > 0 for i in namespaces.values())
        and type(identity) is list
        and len(identity) == 5
        and all(type(i) is int and i >= 0 for i in identity)
        and identity[2] == 0,
        "BOOTSTRAP_IDENTITY_REJECTED",
    )
    mountinfo = stream.read(header["mountinfo_bytes"])
    require(len(mountinfo) == header["mountinfo_bytes"], "BOOTSTRAP_FRAME_TRUNCATED")
    launch["_mount_view"](mountinfo, socket)
    mount_flags = {
        row.split()[4].decode("ascii"): set(row.split()[5].split(b","))
        for row in mountinfo.splitlines()
    }
    require(
        set(mount_flags) == {"/", "/proc", socket, "/dev/null"},
        "BOOTSTRAP_MOUNT_SET_REJECTED",
    )
    require(
        {b"ro", b"nosuid", b"noexec"} <= mount_flags["/dev/null"]
        and {b"ro", b"nosuid", b"nodev", b"noexec"} <= mount_flags[socket],
        "BOOTSTRAP_BIND_FLAGS_REJECTED",
    )
    return header, mountinfo


def candidate(
    manifest: dict[str, Any],
    aliases: dict[str, str],
    header: dict[str, Any],
    mountinfo: bytes,
    contract: dict[str, Any],
    launch: dict[str, Any],
) -> bytes:
    return bytes(
        contract["canonical"](
            {
                "profile": launch["LAUNCH_AUTHORIZATION"],
                "receipt_sha256": manifest["receipt_sha256"],
                "approval_sha256": manifest["approval_sha256"],
                "mountinfo_sha256": contract["digest"](mountinfo),
                "namespaces": header["namespaces"],
                "apparmor_label": header["apparmor_label"],
                "libvirt_socket": manifest["libvirt_socket"],
                "libvirt_socket_identity": header["socket_identity"],
                "libvirt_config_sha256": contract["digest"](CLIENT_BYTES),
                "abi_request_path": ABI_REQUEST,
                "aliases": aliases,
                "data_paths": sorted(manifest["data_files"]),
            }
        )
    )


def verify_review(
    document: bytes,
    key: bytes,
    expected: dict[str, Any],
    scratch: Path,
    contract: dict[str, Any],
    *,
    profile: str = REVIEW,
    decision: str = "ACCEPT_MEASURED_VIEW",
) -> None:
    """The reviewer signs the exact live candidate and evidence using an external key."""
    envelope = contract["loads"](document)
    require(set(envelope) == {"review", "signature_hex"}, "REVIEW_ENVELOPE_REJECTED")
    review = envelope["review"]
    require(
        type(review) is dict and review == {"profile": profile, "decision": decision, **expected},
        "REVIEW_SCOPE_REJECTED",
    )
    signature = envelope["signature_hex"]
    require(
        type(signature) is str
        and len(signature) == 128
        and contract["sha256_value"](signature[:64])
        and contract["sha256_value"](signature[64:])
        and key.startswith(ED25519_SPKI_PREFIX)
        and len(key) == len(ED25519_SPKI_PREFIX) + 32,
        "REVIEW_SIGNATURE_SCHEMA_REJECTED",
    )
    with tempfile.TemporaryDirectory(prefix="review-", dir=scratch) as directory:
        work = Path(directory)
        write(work / "key.der", key, 0o600)
        write(work / "review.json", contract["canonical"](review), 0o600)
        write(work / "signature", bytes.fromhex(signature), 0o600)
        result = subprocess.run(
            [
                "/usr/bin/openssl",
                "pkeyutl",
                "-verify",
                "-pubin",
                "-keyform",
                "DER",
                "-inkey",
                "key.der",
                "-rawin",
                "-in",
                "review.json",
                "-sigfile",
                "signature",
            ],
            cwd=work,
            env=COMMAND_ENV,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=10,
            check=False,
        )
    require(result.returncode == 0, "REVIEW_SIGNATURE_REJECTED")


def apparmor_command(
    manifest: dict[str, Any], action: str, *, diagnostic: bool = False
) -> list[str]:
    policy = Path(__file__).with_name("stage9_v5d11_policy.py")
    receipt = str(Path(manifest["root"]) / "ENVIRONMENT.json")
    return [
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(policy),
        action,
        "--receipt",
        receipt,
        "--receipt-sha256",
        manifest["receipt_sha256"],
        "--abi-request",
        manifest["abi_request"],
        "--socket",
        manifest["libvirt_socket"],
        "--source-sha256",
        manifest["apparmor_source_sha256"],
        "--compiled-sha256",
        manifest["apparmor_compiled_sha256"],
        *(option for name in sorted(manifest["data_files"]) for option in ("--data-path", name)),
        *(["--diagnostic"] if diagnostic else []),
    ]


def seccomp_bytes(bootstrap: Path) -> bytes:
    result = subprocess.run(
        [str(bootstrap), "--dump-seccomp"],
        env=COMMAND_ENV,
        cwd="/",
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=10,
        check=False,
    )
    require(result.returncode == 0 and bool(result.stdout), "SECCOMP_POLICY_DUMP_REFUSED")
    return result.stdout


def build(output: Path) -> dict[str, str]:
    """Compile the inert, statically linked worker and report its pins."""
    require(sys.platform == "linux" and output.is_absolute(), "LINUX_ABSOLUTE_BUILD_PATH_REQUIRED")
    sources = Path(__file__).parent.absolute()
    with tempfile.TemporaryDirectory(prefix="stage9-v5d11-build-") as directory:
        binary = Path(directory) / "bootstrap"
        subprocess.run(
            [
                "/usr/bin/cc",
                "-std=c17",
                "-O2",
                "-D_FORTIFY_SOURCE=2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-static",
                "-o",
                str(binary),
                str(sources / "stage9_v5d11_bootstrap.c"),
                str(sources / "stage9_v5d11_policy.c"),
            ],
            env=COMMAND_ENV,
            cwd=directory,
            stdin=subprocess.DEVNULL,
            timeout=90,
            check=True,
        )
        policy = seccomp_bytes(binary)
        raw = binary.read_bytes()
        write(output, raw, 0o755)
    return {
        "bootstrap_sha256": hashlib.sha256(raw).hexdigest(),
        "seccomp_sha256": hashlib.sha256(policy).hexdigest(),
    }


def check_policy(
    manifest: dict[str, Any], bootstrap: Path, contract: dict[str, Any], *, diagnostic: bool = False
) -> dict[str, Any]:
    result = subprocess.run(
        apparmor_command(manifest, "check", diagnostic=diagnostic),
        env=COMMAND_ENV,
        cwd="/",
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=60,
        check=False,
    )
    require(result.returncode == 0, "APPARMOR_POLICY_NOT_VERIFIED")
    apparmor = contract["loads"](result.stdout)
    require(
        contract["digest"](seccomp_bytes(bootstrap)) == manifest["seccomp_sha256"],
        "SECCOMP_POLICY_PIN_MISMATCH",
    )
    return {"apparmor": apparmor, "seccomp_sha256": manifest["seccomp_sha256"]}


def operator_inputs(
    manifest_path: Path,
) -> tuple[bytes, dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(
        sys.platform == "linux"
        and getattr(os, "geteuid", lambda: -1)() == 0
        and sys.flags.isolated == 1
        and sys.flags.no_site == 1
        and sys.dont_write_bytecode,
        "ISOLATED_LINUX_ROOT_OPERATOR_REQUIRED",
    )
    protected(Path(__file__), private=False)
    scripts = Path(__file__).parent
    trusted = (
        "__init__.py",
        "stage9_v5d10_abi.py",
        "stage9_v5d10_contract.py",
        "stage9_v5d10_launch.py",
        "stage9_v5d11_capture.py",
        "stage9_v5d11_diagnostic.py",
        "stage9_v5d11_images.py",
        "stage9_v5d11_libvirt.py",
        "stage9_v5d11_policy.py",
        "stage9_v5d11_qualify.py",
        "stage9_v5d11_runtime.py",
    )
    for name in trusted:
        protected(scripts / name)
    protected(scripts.parent / "deployment/stage9_v5d11/observer-profile-a.apparmor")
    protected(manifest_path, private=True)
    contract, launch = tool("v5d10_contract"), tool("v5d10_launch")
    raw = cast(bytes, contract["secure_read"](manifest_path, limit=LIMIT))
    return raw, read_manifest(raw, contract, launch), contract, launch


def load_policy(manifest_path: Path, spec_path: Path | None = None) -> None:
    raw, manifest, contract, _ = operator_inputs(manifest_path)
    if spec_path is not None:
        _, spec = diagnostic_inputs(spec_path, manifest_path, raw, manifest, contract)
        manifest = diagnostic_policy(manifest, spec)
    result = subprocess.run(
        apparmor_command(manifest, "load", diagnostic=spec_path is not None),
        env=COMMAND_ENV,
        cwd="/",
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    require(result.returncode == 0, "APPARMOR_POLICY_LOAD_REFUSED")


def load_runtime(
    manifest: dict[str, Any], contract: dict[str, Any]
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    runtime = tool("v5d11_runtime")
    receipt = runtime["candidate_from_manifest"](manifest, contract)
    return runtime, bytes(contract["canonical"](receipt)), receipt


@contextmanager
def staged(
    manifest: dict[str, Any], contract: dict[str, Any], launch: dict[str, Any]
) -> Iterator[tuple[Any, ...]]:
    scratch_base = Path(manifest.get("scratch", "/run"))
    if "scratch" in manifest:
        protected(scratch_base, private=True, directory=True)
    image_paths = any(name.startswith(DATA_PREFIXES[1]) for name in manifest["data_files"])
    images_type = tool("v5d11_images")["ImageStaging"]
    with (
        tempfile.TemporaryDirectory(prefix="stage9-v5d11-", dir=scratch_base) as directory,
        images_type() as images,
    ):
        scratch = Path(directory)
        scratch.chmod(0o700)
        tree = scratch / "root"
        tree.mkdir(mode=0o755)
        tree.chmod(0o755)
        aliases, identity, receipt = (
            stage(tree, manifest, contract, launch, images=images)
            if image_paths
            else stage(tree, manifest, contract, launch)
        )
        runtime, raw, document = load_runtime(manifest, contract)
        runtime_pin = contract["digest"](raw)
        runtime["apply_tree"](tree, document, runtime_pin, contract)
        runtime["verify_tree"](tree, document, runtime_pin, contract)
        yield tree, aliases, identity, receipt, runtime_pin, raw, images


@contextmanager
def isolated_worker(
    tree: Path,
    manifest: dict[str, Any],
    bootstrap: Path,
    expected_identity: tuple[int, ...],
    contract: dict[str, Any],
    launch: dict[str, Any],
    image_args: list[str],
    *,
    diagnostic: bool = False,
) -> Iterator[tuple[subprocess.Popen[bytes], dict[str, Any], bytes]]:
    args = [
        str(bootstrap),
        str(tree),
        str(manifest["uid"]),
        str(manifest["gid"]),
        manifest["libvirt_socket"],
        manifest["root"],
        manifest["receipt_sha256"],
        manifest["approval_sha256"],
        AUTHORITY,
        *(("--diagnostic", "all") if diagnostic else ()),
        *image_args,
    ]
    with subprocess.Popen(
        args,
        env=COMMAND_ENV,
        cwd="/",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        assert (
            process.stdin is not None and process.stdout is not None and process.stderr is not None
        )
        try:
            ready, _, _ = select.select([process.stdout, process.stderr], [], [], 30)
            require(bool(ready), "BOOTSTRAP_MEASUREMENT_TIMEOUT")
            if process.stderr in ready:
                failure = os.read(process.stderr.fileno(), 128).strip().decode("ascii")
                if failure.startswith("BOOTSTRAP_") and failure.replace("_", "").isalnum():
                    raise Refusal(failure)
                raise Refusal("BOOTSTRAP_MEASUREMENT_FAILED")
            header, mountinfo = read_frame(
                process.stdout,
                manifest["libvirt_socket"],
                contract,
                launch,
                diagnostic=diagnostic,
                images=bool(image_args),
            )
            require(process.poll() is None, "LIVE_VIEW_LOST")
            require(
                tuple(header["socket_identity"]) == expected_identity,
                "BOUND_SOCKET_IDENTITY_CHANGED",
            )
            require(
                all(
                    Path(f"/proc/self/ns/{name}").stat().st_ino != value
                    for name, value in header["namespaces"].items()
                ),
                "BOOTSTRAP_NAMESPACE_NOT_PRIVATE",
            )
            yield process, header, mountinfo
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)


def trusted_bootstrap(manifest: dict[str, Any], contract: dict[str, Any]) -> Path:
    bootstrap = Path(manifest["bootstrap"])
    protected(bootstrap)
    pinned(bootstrap, manifest["bootstrap_sha256"], contract)
    return bootstrap


def review_key(manifest: dict[str, Any], contract: dict[str, Any]) -> bytes:
    key_path = Path(manifest["review_key"])
    protected(key_path, private=True)
    return pinned(key_path, manifest["review_key_sha256"], contract)


def measure(manifest_path: Path, output: Path, keep_live: bool = False) -> None:
    raw, manifest, contract, launch = operator_inputs(manifest_path)
    protected(output.parent, private=True, directory=True)
    bootstrap = trusted_bootstrap(manifest, contract)
    policy = check_policy(manifest, bootstrap, contract)
    output.mkdir(mode=0o700)
    output.chmod(0o700)
    with staged(manifest, contract, launch) as staged_view:
        tree, aliases, identity, receipt, runtime_pin, runtime_raw, images = staged_view
        with isolated_worker(
            tree, manifest, bootstrap, identity, contract, launch, images.arguments()
        ) as worker:
            process, header, mountinfo = worker
            require(check_policy(manifest, bootstrap, contract) == policy, "KERNEL_POLICY_CHANGED")
            proposal = candidate(manifest, aliases, header, mountinfo, contract, launch)
            write(output / "runtime-receipt.candidate.json", runtime_raw, 0o600)
            binding = publish(
                output,
                manifest,
                raw,
                receipt,
                policy,
                proposal,
                mountinfo,
                contract,
                runtime_pin=runtime_pin,
                images=images.evidence(),
            )
            if keep_live:
                # Retain the previous API for callers which explicitly request a detached review.
                authority = tree / AUTHORITY[1:]
                with authority.open("r+b") as writer:
                    require(os.fstat(writer.fileno()).st_size == 0, "AUTHORITY_ALREADY_STAGED")
                    writer.write(proposal)
                review = await_review(
                    process,
                    authority,
                    review_key(manifest, contract),
                    binding,
                    tree.parent,
                    contract,
                )
                if review is not None:
                    require(
                        check_policy(manifest, bootstrap, contract) == policy,
                        "KERNEL_POLICY_CHANGED",
                    )
                    write(output / "review.json", review, 0o600)
            assert process.stdin is not None
            process.stdin.close()
            require(process.wait(timeout=10) == 0, "BOOTSTRAP_CANCEL_FAILED")


def publish(
    output: Path,
    manifest: dict[str, Any],
    raw: bytes,
    receipt: dict[str, Any],
    policy: dict[str, Any],
    proposal: bytes,
    mountinfo: bytes,
    contract: dict[str, Any],
    *,
    runtime_pin: str | None = None,
    images: dict[str, Any] | None = None,
) -> dict[str, str]:
    session = secrets.token_hex(32)
    candidate_pin = contract["digest"](proposal)
    record = contract["canonical"](
        {
            "profile": EVIDENCE,
            "status": "CANDIDATE_ONLY",
            "session": session,
            "authorization": "NOT_AUTHORIZED",
            "native_load": "NOT_TESTED",
            "manifest_sha256": contract["digest"](raw),
            "candidate_sha256": candidate_pin,
            "mountinfo_sha256": contract["digest"](mountinfo),
            "file_plan_sha256": receipt["file_plan_sha256"],
            "policy": policy,
            "bootstrap_sha256": manifest["bootstrap_sha256"],
            "review_key_sha256": manifest["review_key_sha256"],
            "data_sha256": {name: row["sha256"] for name, row in manifest["data_files"].items()},
            "runtime_receipt_sha256": runtime_pin,
            "immutable_images": images or {},
            "release_blockers": ["NATIVE_QUALIFICATION_NOT_TESTED"],
        }
    )
    write(output / "mountinfo", mountinfo, 0o600)
    write(output / "launch-authorization.candidate.json", proposal, 0o600)
    write(output / "measurement.json", record, 0o600)
    print('{"status":"CANDIDATE_ONLY","authorization":"NOT_AUTHORIZED"}', flush=True)
    return {
        "session": session,
        "candidate_sha256": candidate_pin,
        "measurement_sha256": contract["digest"](record),
    }


def await_review(
    process: subprocess.Popen[bytes],
    authority: Path,
    key: bytes,
    expected: dict[str, str],
    scratch: Path,
    contract: dict[str, Any],
) -> bytes | None:
    """Wait for a reviewer-supplied file path while this exact view stays alive."""
    assert process.stdout is not None
    ready, _, _ = select.select([sys.stdin, process.stdout], [], [], CONTROL_TIMEOUT)
    require(
        sys.stdin in ready and process.stdout not in ready and process.poll() is None,
        "LIVE_VIEW_LOST",
    )
    path_text = sys.stdin.readline(4096).rstrip("\n")
    if not path_text:
        return None
    require(contract["path_value"](path_text, absolute=True), "REVIEW_PATH_REJECTED")
    path = Path(path_text)
    protected(path, private=True)
    raw = cast(bytes, contract["secure_read"](path, limit=LIMIT))
    verify_review(raw, key, expected, scratch, contract)
    pinned(authority, expected["candidate_sha256"], contract)
    require(process.poll() is None, "LIVE_VIEW_LOST")
    return raw


def private_document(path: Path, contract: dict[str, Any]) -> bytes:
    protected(path, private=True)
    return cast(bytes, contract["secure_read"](path, limit=LIMIT))


def diagnostic_inputs(
    path: Path,
    manifest_path: Path,
    manifest_raw: bytes,
    manifest: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    raw = private_document(path, contract)
    doc: dict[str, Any] = contract["loads"](raw)
    paths = {"manifest", "measurement", "static_report", "fixture_socket"}
    pins = {
        "manifest_sha256",
        "measurement_sha256",
        "runtime_receipt_sha256",
        "static_report_sha256",
        "payload_sha256",
        "apparmor_source_sha256",
        "apparmor_compiled_sha256",
        "readonly_implementation_sha256",
    }
    require(
        set(doc)
        == {"profile", "domain_uuid", "network_name", "fixture_pid", "fixtures", *paths, *pins}
        and doc["profile"] == DIAGNOSTIC_SPEC
        and all(contract["path_value"](doc[key], absolute=True) for key in paths)
        and all(contract["sha256_value"](doc[key]) for key in pins)
        and doc["manifest"] == str(manifest_path)
        and doc["manifest_sha256"] == contract["digest"](manifest_raw)
        and doc["fixture_socket"]
        not in {"/run/libvirt/libvirt-sock-ro", "/run/libvirt/virtqemud-sock-ro"}
        and type(doc["fixture_pid"]) is int
        and doc["fixture_pid"] > 1,
        "DIAGNOSTIC_INPUTS_REJECTED",
    )
    require(
        type(doc["domain_uuid"]) is str
        and str(uuid.UUID(doc["domain_uuid"])) == doc["domain_uuid"]
        and type(doc["network_name"]) is str
        and 0 < len(doc["network_name"]) <= 128
        and type(doc["fixtures"]) is dict
        and set(doc["fixtures"]) == {"raw", "qcow2"}
        and all(
            type(item) is str
            and item.startswith(DATA_PREFIXES[1])
            and item in manifest["data_files"]
            for item in doc["fixtures"].values()
        )
        and len(set(doc["fixtures"].values())) == 2,
        "DIAGNOSTIC_FIXTURE_REJECTED",
    )
    measurement_path = Path(doc["measurement"])
    protected(measurement_path, private=True)
    measured = contract["loads"](pinned(measurement_path, doc["measurement_sha256"], contract))
    require(
        measured.get("profile") == EVIDENCE
        and measured.get("status") == "CANDIDATE_ONLY"
        and measured.get("manifest_sha256") == doc["manifest_sha256"]
        and measured.get("runtime_receipt_sha256") == doc["runtime_receipt_sha256"]
        and set(measured.get("immutable_images", {})) >= set(doc["fixtures"].values()),
        "DIAGNOSTIC_MEASUREMENT_BINDING_REJECTED",
    )
    return raw, doc


def diagnostic_policy(manifest: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    keys = ("apparmor_source_sha256", "apparmor_compiled_sha256")
    return {**manifest, **{key: spec[key] for key in keys}}


def diagnostic_authority(
    authorization: Path,
    spec_raw: bytes,
    manifest: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[bytes, str]:
    raw = private_document(authorization, contract)
    review = contract["loads"](raw).get("review")
    require(
        type(review) is dict
        and contract["sha256_value"](review.get("nonce"))
        and type(review.get("expires_utc")) is int
        and time.time() < review["expires_utc"] <= time.time() + 3600,
        "DIAGNOSTIC_AUTHORIZATION_WINDOW_REJECTED",
    )
    verify_review(
        raw,
        review_key(manifest, contract),
        {
            "spec_sha256": contract["digest"](spec_raw),
            "nonce": review["nonce"],
            "expires_utc": review["expires_utc"],
        },
        authorization.parent,
        contract,
        profile=DIAGNOSTIC_REVIEW,
        decision="AUTHORIZE_NATIVE_DIAGNOSTIC_ONLY",
    )
    return raw, review["nonce"]


def consume_diagnostic_authority(path: Path, approval: bytes) -> None:
    write(path, approval, 0o600)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    flags = getattr(os, "O_DIRECTORY", None)
    require(type(flags) is int, "OPERATOR_DIRECTORY_SYNC_UNAVAILABLE")
    assert flags is not None
    directory = os.open(path.parent, os.O_RDONLY | flags)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def check_dedicated_account(uid: int, gid: int, proc: Path = Path("/proc")) -> None:
    """Refuse credentials shared with any existing host process."""
    for entry in proc.iterdir():
        if not (entry.name.isascii() and entry.name.isdecimal()):
            continue
        try:
            raw = (entry / "status").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        rows = {
            name: value.split()
            for name, _, value in (line.partition(b":") for line in raw.splitlines())
        }
        require(
            not ({str(uid).encode()} & set(rows[b"Uid"]))
            and not ({str(gid).encode()} & set(rows[b"Gid"] + rows[b"Groups"])),
            "DIAGNOSTIC_ACCOUNT_IN_USE",
        )


def stage_diagnostic(
    tree: Path,
    manifest: dict[str, Any],
    spec: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    payload = Path(__file__).with_name("stage9_v5d11_diagnostic.py")
    protected(payload)
    raw = pinned(payload, spec["payload_sha256"], contract)
    config = {
        "root": manifest["root"],
        "socket": manifest["libvirt_socket"],
        "domain_uuid": spec["domain_uuid"],
        "network_name": spec["network_name"],
        "images": {
            name: {"path": path, "sha256": manifest["data_files"][path]["sha256"]}
            for name, path in spec["fixtures"].items()
        },
    }
    set_owner = getattr(os, "chown", None)
    require(callable(set_owner), "POSIX_FILE_OWNER_REQUIRED")
    assert set_owner is not None
    for name, data in (
        (DIAGNOSTIC_SCRIPT, raw),
        (DIAGNOSTIC_CONFIG, contract["canonical"](config)),
    ):
        target = tree / name[1:]
        write(target, data, 0o440)
        set_owner(target, 0, manifest["gid"])


def diagnose(manifest_path: Path, spec_path: Path, authorization: Path, output: Path) -> None:
    manifest_raw, manifest, contract, launch = operator_inputs(manifest_path)
    protected(output.parent, private=True, directory=True)
    protected(manifest_path.parent, private=True, directory=True)
    spec_raw, spec = diagnostic_inputs(spec_path, manifest_path, manifest_raw, manifest, contract)
    approval, nonce = diagnostic_authority(authorization, spec_raw, manifest, contract)
    bootstrap = trusted_bootstrap(manifest, contract)
    diagnostic_manifest = diagnostic_policy(manifest, spec)
    policy = check_policy(diagnostic_manifest, bootstrap, contract, diagnostic=True)
    report = contract["loads"](
        pinned(spec["static_report"], spec["static_report_sha256"], contract, limit=32 * LIMIT)
    )
    # The trusted, protected checkout contains the host-only collector imports.
    checkout = str(Path(__file__).absolute().parents[1])
    sys.path.insert(0, checkout)
    try:
        capture = tool("v5d11_capture")
        qualifier = tool("v5d11_qualify")
        virt = tool("v5d11_libvirt")
    finally:
        sys.path.remove(checkout)
    check_dedicated_account(manifest["uid"], manifest["gid"])
    output.mkdir(mode=0o700)
    output.chmod(0o700)
    with staged(manifest, contract, launch) as staged_view:
        tree, _, identity, receipt, runtime_pin, runtime_raw, images = staged_view
        require(runtime_pin == spec["runtime_receipt_sha256"], "DIAGNOSTIC_RUNTIME_PIN_MISMATCH")
        request = contract["loads"](
            pinned(manifest["abi_request"], receipt["abi_request_sha256"], contract)
        )
        qualifier["_objects"](report, receipt, request)
        stage_diagnostic(tree, manifest, spec, contract)
        with isolated_worker(
            tree,
            manifest,
            bootstrap,
            identity,
            contract,
            launch,
            images.arguments(),
            diagnostic=True,
        ) as worker:
            process, header, mountinfo = worker
            require(
                check_policy(diagnostic_manifest, bootstrap, contract, diagnostic=True) == policy,
                "KERNEL_POLICY_CHANGED",
            )
            # Consume the signed nonce with an exclusive inode before a native action.
            require(
                time.time() < contract["loads"](approval)["review"]["expires_utc"],
                "DIAGNOSTIC_AUTHORIZATION_EXPIRED",
            )
            consume_diagnostic_authority(
                manifest_path.parent / f"diagnostic-used-{nonce}", approval
            )
            host_before = virt["measure_host"](
                manifest["libvirt_socket"], spec["readonly_implementation_sha256"]
            )
            require(
                tuple(host_before["host_socket_identity"]) == identity,
                "DIAGNOSTIC_SOCKET_IDENTITY_CHANGED",
            )
            denial = virt["collect_fixture_denial"](
                Path(spec["fixture_socket"]),
                spec["fixture_pid"],
                spec["domain_uuid"],
                host_before,
            )
            write(output / "mountinfo", mountinfo, 0o600)
            write(output / "runtime-receipt.json", runtime_raw, 0o600)
            view = {
                "mountinfo": {"path": "mountinfo", "sha256": contract["digest"](mountinfo)},
                "namespaces": header["namespaces"],
                "apparmor_sha256": spec["apparmor_compiled_sha256"],
                "seccomp_sha256": manifest["seccomp_sha256"],
                "purpose": "diagnostic",
            }
            loader, libvirt = capture["collect"](
                process,
                header["host_pid"],
                view,
                receipt,
                request,
                report,
                output,
                staging_root=tree,
                on_libvirt=lambda pid: virt["observe_live_connections"](
                    pid, manifest["libvirt_socket"]
                ),
            )
            require(process.wait(timeout=10) == 0, "DIAGNOSTIC_EXECUTION_FAILED")
            host_after = virt["measure_host"](
                manifest["libvirt_socket"], spec["readonly_implementation_sha256"]
            )
            require(
                check_policy(diagnostic_manifest, bootstrap, contract, diagnostic=True) == policy,
                "KERNEL_POLICY_CHANGED",
            )
            native = {
                "profile": qualifier["PROFILE"],
                "receipt_sha256": manifest["receipt_sha256"],
                "abi_request_sha256": receipt["abi_request_sha256"],
                "static_report_sha256": spec["static_report_sha256"],
                "view": view,
                "loader": loader,
                "libvirt": virt["assemble"](
                    host_before,
                    host_after,
                    libvirt["view"],
                    libvirt["read_results"],
                    libvirt["connect_paths"],
                    denial,
                ),
            }
            assessment = qualifier["assess"](
                receipt,
                request,
                report,
                native,
                receipt_sha256=manifest["receipt_sha256"],
                static_report_sha256=spec["static_report_sha256"],
                artifacts=output,
            )
            write(output / "native-evidence.json", contract["canonical"](native), 0o600)
            write(output / "assessment.json", contract["canonical"](assessment), 0o600)
            write(
                output / "diagnostic-run.json",
                contract["canonical"](
                    {
                        "profile": DIAGNOSTIC_SPEC,
                        "spec_sha256": contract["digest"](spec_raw),
                        "authorization_sha256": contract["digest"](approval),
                        "runtime_receipt_sha256": runtime_pin,
                        "evidence_sha256": contract["digest"](contract["canonical"](native)),
                        "immutable_images": images.evidence(),
                        "policy": policy,
                    }
                ),
                0o600,
            )
    print(
        '{"status":"EVIDENCE_CONSISTENT_REVIEW_REQUIRED","authorization":"NOT_GRANTED"}', flush=True
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("measure")
    collect.add_argument("manifest", type=Path)
    collect.add_argument("output", type=Path)
    collect.add_argument("--wait-for-review", action="store_true")
    load = commands.add_parser("load-policy")
    load.add_argument("manifest", type=Path)
    load.add_argument("--diagnostic-inputs", type=Path)
    diagnostic = commands.add_parser("diagnose")
    for name in ("manifest", "spec", "authorization", "output"):
        diagnostic.add_argument(name, type=Path)
    commands.add_parser("build").add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "load-policy":
            load_policy(args.manifest, args.diagnostic_inputs)
        elif args.command == "build":
            print(json.dumps(build(args.output), sort_keys=True))
        elif args.command == "diagnose":
            diagnose(args.manifest, args.spec, args.authorization, args.output)
        else:
            measure(args.manifest, args.output, args.wait_for_review)
    except Refusal as exc:
        print(f"REFUSED {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError):
        print("REFUSED OPERATOR_INPUT_OR_HOST_REJECTED", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
